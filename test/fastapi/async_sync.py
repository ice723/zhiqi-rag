import threading

import uvicorn
from fastapi import FastAPI

app = FastAPI()

#异步请求
@app.get("/simple-async")
async def simple_async():
    print(threading.current_thread().ident) #查询线程号
    return {"message": "Hello async"}

#同步请求
@app.get("/simple-sync")
def simple_sync():
    print(threading.current_thread().ident)
    return {"message": "Hello sync"}

#fastapi的调用
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

