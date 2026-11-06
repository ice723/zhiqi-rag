from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class MineruConfig:
    base_url : str
    api_token : str
mineru_config = MineruConfig(
    base_url=os.getenv("MINERU_BASE_URL",""),  #s.getenv,从系统 /`.env`环境变量读取对应的值；如果找不到这个环境变量，就返回第二个参数`""`空字符串
    api_token=os.getenv("MINERU_API_TOKEN","")
)
