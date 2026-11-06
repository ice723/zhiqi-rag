import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

#cors 准许跨域协议
app.add_middleware(
    CORSMiddleware, #跨域中间件，让浏览器允许这个跨域请求
    allow_origins=["*"], #允许所有的源
    allow_credentials=True,#允许客户端传递cookie
    allow_methods=["*"],#get跟post都行
    allow_headers=["*"],#请求头所有信息都行
)
@app.get("/fast_api")
async def root():
    return {"message": "Hello World"}
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)