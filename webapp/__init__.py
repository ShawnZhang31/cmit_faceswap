import os
from flask import Flask
from config import config
from faceswap.facelib import GenderToolClass
import dlib

GENDER_PROTOTXT_FILE_PATH=os.getenv("GENDER_PROTOTXT_FILE_PATH") or "./res/gender/gender_deploy.prototxt"
GENDER_NET_FILE_PATH=os.getenv("GENDER_NET_FILE_PATH") or "./res/gender/gender_net.caffemodel"
DLIB_FACE_LANDMARK_SHAPE_FILE_PATH=os.getenv("DLIB_FACE_LANDMARK_SHAPE_FILE_PATH") or "./res/dlib/shape_predictor_68_face_landmarks.dat"

# print(GENDER_PROTOTXT_FILE_PATH)

gender_classifer = GenderToolClass(proto_file_path=GENDER_PROTOTXT_FILE_PATH, model_file_path=GENDER_NET_FILE_PATH)
dlib_landmark_predictor = dlib.shape_predictor(DLIB_FACE_LANDMARK_SHAPE_FILE_PATH)

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # 注册路由和自定义的错误页面
    from webapp.main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app

