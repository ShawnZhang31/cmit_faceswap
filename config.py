"""
应用配置文件
"""

import os

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "hard to guess string"

    @staticmethod
    def init_app(app):
        pass

# 开发环境配置
class DevelopmentConfig(Config):
    DEBUG = True

# 测试环境配置
class TestingConfig(Config):
    TESTING = True

# 生产环境配置
class ProductionConfig(Config):
    DEBUG = False
    pass

config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,

    "default": DevelopmentConfig
}
