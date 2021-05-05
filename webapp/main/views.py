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
from faceswap.facelib.mediaface import getFaces, getFaceLandmarks, faceSwap

from faceswap.faceswap_utils.ImageException import ImageError
from faceswap.faceswap_utils.ModelLoadException import ModelLoadError




@main.route("/", methods=['GET', 'POST'])
def index():
    # add = Add(3, 2)
    return "Hello Face Swap from CMIT"
    # return json.dumps({"test":add.sum()}), 200

# 人脸融合V1
@main.route("/api/v1/faceswap", methods=['POST'])
def faceswap_v1():
    # 获取提交表单
    form = request.form
    
    # 提取表单中的img字段
    img_base64 = form.get('image_ref')

    # 读取图片文件
    img1 = imageFromBase64Code(img_base64)
    

    # gender_prototxt_file_path = "./res/gender/gender_deploy.prototxt"
    # gender_net_file_path = "./res/gender/gender_net.caffemodel"

    # dlib_shape_model_path = "./res/dlib/shape_predictor_68_face_landmarks.dat"



    tempalte_path = "./res/template1/"

    # genderClassfier = GenderToolClass(gender_prototxt_file_path, gender_net_file_path)
    # img1_gender = genderClassfier.getGender(img1)
    # 使用blazeface加快合成速度
    faces, bboxs = getFaces(img1)
    img1_gender = gender_classifer.getGender(img1, box=bboxs[0])
    
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

@main.route("/api/v2/faceswap", methods=['POST'])
def faceswap_v2():
    # 获取提交表单
    form = request.form
    
    # 提取表单中的img字段
    img_ref_base64 = form.get('image_ref')

    # 读取图片文件
    img_ref = imageFromBase64Code(img_ref_base64)

    tempalte_path = "./res/template1/"

    # 是因为mediapipe获取图像的面部bbox
    faces, bboxs = getFaces(img_ref)
    img_ref_gender = gender_classifer.getGender(img_ref, box=bboxs[0])
    template_name = "template_F.png"
    if img_ref_gender == 'male':
        template_name = "template_M.png"
    elif img_ref_gender == 'female':
        template_name = "template_F.png"
    else:
        pass
    img_template_path = tempalte_path + template_name

    img_template = cv2.imread(img_template_path, cv2.IMREAD_COLOR)

    # 获取landmarks
    img_tempalte_landmarks = getFaceLandmarks(img_template)[0]
    img_ref_landmarks = getFaceLandmarks(img_ref)[0]

    output = faceSwap(img_template, img_ref, img_tempalte_landmarks, img_ref_landmarks)

    
    base64_string = base64EncodeImage(output)


    return json.dumps({'success':True, 'result': base64_string}), 200, {'ContentType':'application/json'} 
