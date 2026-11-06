#常见的输出方式
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import RedirectResponse

app = FastAPI()
@app.get("/item/{item_id}")
async def read_item(item_id: int, a: str):
    print("read_item后端有参数接被访问")
    return {"item_id": item_id, "a": a}

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None

@app.post("/item")
def create_item(item: Item):
    print("create_item")
    print("create_item后端接口被访问")
    return item

@app.get("/old-path")
async def redirect_old_path():
    return RedirectResponse(url="/old-path", status_code=307)


