""" 导入流程节点基类
定义统一的节点接口规范，提供通用功能
"""
from abc import ABC, abstractmethod
from typing import TypeVar
import logging

T = TypeVar("T")  # 泛型状态类型

class BaseNode(ABC):
    """
    导入流程节点基类
    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、错误处理。
    """
    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self):
        """初始化节点"""
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口
        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出和异常处理。
        """
        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise e

    @abstractmethod
    def process(self, state: T) -> T:
        """子类必须实现核心逻辑"""
        pass

    def log_step(self, step_name: str, message: str = ""):
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
