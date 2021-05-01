# 路由
from . import main

@main.route("/", methods=['GET', 'POST'])
def index():
    return "Hello Face Swap from CMIT"

@main.route("/faceswap/v1", methods=['GET'])
def faceswap_v1():
    return "Hello Face Swap v1"