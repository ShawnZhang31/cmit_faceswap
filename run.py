# 应用脚本
import os
from webapp import create_app

# print(os.getenv("APP_CONFIG") or "default")

app = create_app(os.getenv("APP_CONFIG") or "default")
