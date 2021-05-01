from flask import Flask

from config import config

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # 注册路由和自定义的错误页面
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
