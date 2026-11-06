# processor/import_processor/nodes/node_md_img.py
import base64
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, Dict, List, Deque, Optional

from minio import Minio
from minio.deleteobjects import DeleteObject

from config.minio_config import minio_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from utils.llm_utils import get_vl_client
from utils.minio_utils import build_public_url, get_minio_client


class NodeMDImg(BaseNode):
    """
    MarkDown图片处理节点：多模态图片理解
    """
    name = "node_md_img"

    def __init__(self):
        super().__init__()
        from processor.import_processor.import_config import get_config
        self.config = get_config()

    def process(self, state: ImportGraphState):
        self.logger.info(f"{self.name}节点开始执行...")

        #1.参数处理
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)

        #2.图片扫描
        target_images = self._step_2_scan_images(md_content, images_dir)
        self.logger.info(f"扫描到待处理图片数量: {len(target_images)}")
        if not target_images:
            self.logger.info("没有找到需要处理的图片，直接返回原始内容")
            state["md_content"] = md_content
            return state

        #3.视觉模型摘要
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)

        #4.上传minio，替换md
        new_md_content = self._step_4_upload_and_replace(md_path_obj.stem, target_images, summaries, md_content)

        #5.保存备份md
        new_md_file_name = self._step_5_backup_new_md_file(state["md_path"], new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name
        return state

    #步骤1
    def _step_1_get_content(self, state: ImportGraphState):
        #1.校验参数
        md_path = state.get("md_path")
        if not md_path:
            raise StateFieldError("md_path", expected_type=str)
        md_path_obj = Path(md_path)
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"输入文件不存在：{md_path}")

        md_content = state.get("md_content")
        #测试代码
        if not md_content:
            with open(md_path, "r", encoding="utf-8") as md_file:
                md_content = md_file.read()

        #2.图片路径，和md文档所在目录同级的images图片路径
        images_dir = md_path_obj.parent / "images"

        return md_content, md_path_obj, images_dir

    #步骤2
    def _step_2_scan_images(self, md_content, images_dir: Path):
        """
        找到图片并截取上下文
        """
        #1.返回结果
        target_images = []

        # images目录不存在直接返回空列表
        if not images_dir.exists() or not images_dir.is_dir():
            self.logger.warning(f"图片目录不存在: {images_dir}")
            return target_images

        #2.扫描图片
        for images_file in os.listdir(images_dir):
            file_ext = os.path.splitext(images_file)[1].lower()
            if file_ext not in self.config.image_extensions:
                self.logger.warning(f"图片格式不支持：{images_file}")
                continue

            img_path = images_dir / images_file  #图片路径
            content = self._find_image_in_md(md_content, images_file)

            #过滤MD中未引用的图片
            if not content:  #当找不到图片的文件名时候，说明图片没有在md文档引用
                self.logger.warning(f"图片未在MD文档中引用：{images_file}")
                continue

            target_images.append((images_file, img_path, content))
        return target_images

    #步骤2的方法1
    def _find_image_in_md(self, md_content: str, images_file: str, content_len: int = 100) -> Optional[Tuple[str, str]]:
        """
        找到图片并获取上下文
        返回 None 代表没有匹配到图片
        """
        partern = re.compile(r"!\[.*?\]\(.*?" + re.escape(images_file) + r".*?\)")
        match = partern.search(md_content)
        if not match:
            return None
        start, end = match.span()

        pre_content = md_content[max(0, start - content_len):start]  #图片的上文内容
        post_text = md_content[end:min(len(md_content), end + content_len)]  #图片的下文内容
        return pre_content, post_text

    #步骤3
    def _step_3_generate_summaries(self, md_stem: str, target_images: List[Tuple[str, Path, Tuple[str, str]]]) -> Dict[str, str]:
        summaries = {}
        # 修复：队列初始化放在循环外，滑动窗口限流生效
        request_deque: Deque[float] = deque()

        #用来动态调度任务
        for img_file, image_path, content in target_images:
            self._apply_api_rate_limit(request_deque, max_requests=10)
            #调用视觉模型
            try:
                summaries[img_file] = self._summarize_image(image_path, root_folder=md_stem, image_content=content)
                self.logger.info(f"图片[{img_file}]摘要生成成功: {summaries[img_file]}")
            except Exception as e:
                self.logger.error(f"图片{img_file}生成摘要失败, skip this img: {str(e)}")
                summaries[img_file] = "图片"
        return summaries

    #步骤3的方法1，任务调度函数
    def _apply_api_rate_limit(self, request_times: Deque[float], max_requests: int, window_seconds: int = 60) -> None:
        """
            通用滑动窗口API速率限制器（抽离为公共工具）
            核心逻辑：维护请求时间戳双端队列，窗口内请求数超上限则自动等待，防止触发第三方API限流
            :param request_times: 存储请求时间戳的双端队列，需外部初始化（全局/单例），跨调用复用
            :param max_requests: 速率限制窗口内的最大允许请求次数
            :param window_seconds: 速率限制滑动窗口时长，默认60秒（1分钟）
            :return: None，超出限制时会阻塞等待
        """
        current_time = time.time()

        #1.清理滑动窗口外的过期请求时间戳，保证队列仅存窗口内的请求
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        #2.窗口内请求数达到上限，计算并阻塞等待时间
        if len(request_times) >= max_requests:
            # 计算需求等待的时长(窗口总时长-最早请求已存在的时长)
            sleep_duration = window_seconds - (current_time - request_times[0])
            if sleep_duration > 0:
                self.logger.info(
                    f"触发API速率限制，窗口{window_seconds}秒内最多{max_requests}次，需等待：{sleep_duration:.2f}秒")
                time.sleep(sleep_duration)
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        #3. 记录当前请求时间戳，加入滑动窗口队列
        request_times.append(current_time)
        self.logger.info(f"API请求已记录，当前{window_seconds}秒窗口内请求数：{len(request_times)}")

    #步骤3方法2
    def _summarize_image(self, image_path: Path, root_folder: str, image_content: Tuple[str, str]) -> str:
        #图片的base64编码
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        #链接视觉模型（DashScope qwen3-vl-plus）
        vi_ai = get_vl_client()

        #调用模型
        pre_text = ""
        post_text = ""
        if image_content:
            pre_text, post_text = image_content
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""这是"{root_folder}"文件中的一张图片，图片上文部分为"{pre_text}"，下文部分为"{post_text}"，
                        请用中文简要总结这张图片的内容，用于 Markdown 图片标题。尽量不要超过10个字"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{self._mime_type(image_path)};base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        response = vi_ai.invoke(messages)

        #3.处理返回摘要信息
        return response.content.strip().replace("\n", "")

    @staticmethod
    def _mime_type(image_path: Path) -> str:
        mapping = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "tif": "image/tiff",
            "tiff": "image/tiff",
            "bmp": "image/bmp",
        }
        return mapping.get(image_path.suffix.lstrip(".").lower(), "image/jpeg")

    def _step_4_upload_and_replace(self, doc_stem: str, target_images: List[Tuple[str, Path, Tuple[str, str]]], summaries: Dict[str, str], md_content: str):
        # 0 minio客户端
        minio_client = get_minio_client()
        minio_img_dir = minio_config.img_dir
        upload_dir = f"{minio_img_dir}/{doc_stem}"
        self.logger.info(f"图片存入{upload_dir}目录下")

        #1.清理minio目录
        self.clean_minio_directory(minio_client, upload_dir)

        #2.批量上传图片，获得minio的urls
        urls = self.upload_images_batch(minio_client, upload_dir, target_images)

        #3.将摘要和url路径合并
        image_info = self.merge_summary_and_url(summaries, urls)

        #4.替换md文档的摘要和路径
        md_content = self.process_md_file(md_content, image_info)

        return md_content

    #步骤4的方法1清理目录
    def clean_minio_directory(self, minio_client: Minio, upload_dir):
        object_to_delete = minio_client.list_objects(minio_config.bucket_name, prefix=upload_dir, recursive=True)
        #构造删除列表
        delete_list = [DeleteObject(obj.object_name) for obj in object_to_delete]
        if delete_list:
            error_results = minio_client.remove_objects(minio_config.bucket_name, delete_list)
            for error in error_results:
                self.logger.error(f"删除失败：{error}")

    #步骤4方法2批量上传文件
    def upload_images_batch(self, minio_client: Minio, upload_dir: str, target_images: List[Tuple[str, Path, Tuple[str, str]]]):
        urls = {}
        for img_file, img_path, _ in target_images:
            try:
                #上传
                object_name = f"{upload_dir}/{img_file}"  #minio文件对象名{路径，带后缀}
                self.logger.info(f"上传文件：{object_name}")
                urls[img_file] = self.upload_to_minio(minio_client, img_path, object_name)  #返回minio的url
            except Exception as e:
                self.logger.error(f"图片{img_file}上传MinIO失败：{str(e)}")
        return urls

    #步骤4方法2上传minio
    def upload_to_minio(self, minio_client: Minio, img_path: Path, object_name: str) -> str:
        suffix = img_path.suffix.lstrip(".")
        # 简单修正常见mimetype
        mime_map = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
            "tif": "image/tiff",
            "tiff": "image/tiff",
            "bmp": "image/bmp"
        }
        content_type = mime_map.get(suffix.lower(), f"image/{suffix}")

        #上传minio
        minio_client.fput_object(
            bucket_name=minio_config.bucket_name,
            object_name=object_name,
            file_path=str(img_path),
            content_type=content_type
        )
        # 使用对外可访问地址（MINIO_PUBLIC_BASE_URL），保证浏览器能加载图片
        return build_public_url(object_name)

    #步骤4方法3合并参数
    def merge_summary_and_url(self, summary: Dict[str, str], urls) -> Dict[str, Tuple[str, str]]:
        image_info = {}
        for img_file, summary_text in summary.items():
            if url := urls.get(img_file):
                image_info[img_file] = (summary_text, url)
        return image_info

    #步骤4方法替换url和摘要summary
    def process_md_file(self, md_content: str, image_info: Dict[str, Tuple[str, str]]):
        for image_file, (summary, url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)
        return md_content

    #步骤5保存和备份新文档
    def _step_5_backup_new_md_file(self, origin_md_path: str, md_content: str) -> str:
        """
        步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
        """
        #构造新文件路径：替换原后缀为 ——new.md
        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

        #写入新的MD内容（覆盖写入，若文件已存在则更新）
        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)
        self.logger.info(f"处理后的MD文件已经保存，新文件路径为：{new_md_file_name}")

        return new_md_file_name


if __name__ == '__main__':
    setup_logging()


    init_state = {
        "md_path": r"D:\output\PantumP3500用户手册zh_CNV1.2_1644316283788\PantumP3500用户手册zh_CNV1.2_1644316283788.md",
        "md_content": None
    }
    node = NodeMDImg()
    result = node(init_state)

    dumps = json.dumps(result, ensure_ascii=False, indent=2)
    print(dumps)
