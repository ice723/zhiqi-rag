import time

import uvicorn
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()
#backgroundTask（后台任务）业务叫法
# 是接口先直接返回响应，任务丢到后面异步后端服务器慢慢跑，不卡前端的请求

def write_log(email: str, content: str):
    while True:
        print(f"正在给{email}邮箱发 {content}消息....")
        time.sleep(1)


@app.get("/send-task/{email}")
async def send_task(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_log, email, "你好")
    print("接口开始")
    time.sleep(1)
    print("接口结束")
    return {"message": "接口执行完成"}
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)