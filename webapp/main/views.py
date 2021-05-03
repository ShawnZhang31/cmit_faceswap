# 路由
from flask import request
from . import main
import base64
import cv2
import numpy as np
import json

from webapp import gender_classifer, dlib_landmark_predictor
from webapp.utils import imageFromBase64Code, base64EncodeImage

from faceswap.facelib import DlibToolClass, GenderToolClass
from faceswap.swap import FaceSwap

from faceswap.faceswap_utils.ImageException import ImageError
from faceswap.faceswap_utils.ModelLoadException import ModelLoadError




@main.route("/", methods=['GET', 'POST'])
def index():
    # add = Add(3, 2)
    return "Hello Face Swap from CMIT"
    # return json.dumps({"test":add.sum()}), 200

# 人脸融合V1
@main.route("/faceswap/v1", methods=['POST'])
def faceswap_v1():
    # 获取提交表单
    form = request.form
    
    # 提取表单中的img字段
    img_base64 = form.get('image')

    # 读取图片文件
    img1 = imageFromBase64Code(img_base64)
    

    # gender_prototxt_file_path = "./res/gender/gender_deploy.prototxt"
    # gender_net_file_path = "./res/gender/gender_net.caffemodel"

    # dlib_shape_model_path = "./res/dlib/shape_predictor_68_face_landmarks.dat"



    tempalte_path = "./res/template1/"

    # genderClassfier = GenderToolClass(gender_prototxt_file_path, gender_net_file_path)
    # img1_gender = genderClassfier.getGender(img1)
    img1_gender = gender_classifer.getGender(img1)
    template_name = "template_F.png"
    if img1_gender == 'male':
        template_name = "template_M.png"
    elif img1_gender == 'female':
        template_name = "template_F.png"
    else:
        pass
    img2_path = tempalte_path + template_name

    img2 = cv2.imread(img2_path)


    swapface = FaceSwap(dlib_landmark_predictor)
    merge = swapface.swap_face(img1, img2)
    

    base64_string = base64EncodeImage(merge)


    return json.dumps({'success':True, 'result': base64_string}), 200, {'ContentType':'application/json'} 
    # return base64_string
