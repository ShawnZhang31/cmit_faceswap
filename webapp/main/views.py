# 路由
import os
from numpy.lib.type_check import imag
import yaml
from flask import request
from . import main
import base64
import cv2
import numpy as np
import json

from webapp import gender_classifer, dlib_landmark_predictor
from webapp.utils import imageFromBase64Code, base64EncodeImage

from faceswap.facelib import DlibToolClass, GenderToolClass, gender
from faceswap.swap import FaceSwap
from faceswap.facelib.mediaface import getFaces, getFaceLandmarks, faceSwap

from faceswap.faceswap_utils.ImageException import ImageError
from faceswap.faceswap_utils.ModelLoadException import ModelLoadError

from webapp.utils.decorators.check_image import image_required_withkey
from webapp.utils.decorators.check_params import request_required_params
from webapp.utils.decorators.check_template import getTemplateImgs, tempate_required_withkey
from webapp.utils.API_RESPONE_CODE import API_RESPONE_CODE




@main.route("/", methods=['GET', 'POST'])
def index():
    # add = Add(3, 2)
    return "Hello Face Swap from CMIT"
    # return json.dumps({"test":add.sum()}), 200

# 人脸融合V1
@main.route("/api/v1/faceswap", methods=['POST'])
@request_required_params(['image_ref', 'template_name'])
@image_required_withkey('image_ref')
@tempate_required_withkey('template_name')
def faceswap_v1(*args, **kwargs):
    image_ref = kwargs['image_ref']
    template = kwargs['template']
    
    resp={}
    resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
    resp['error'] = None
    resp['swaped_image'] = None

    # 检查image_ref中是否有人脸
    faces, bboxes = getFaces(image_ref)
    if len(faces) == 0:
        resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
        resp['error'] = "上传的图像中未检测到人脸"
        resp['swaped_image']=None
        return json.dumps(resp)

    # 检查上传的图像的性别
    image_ref_gender = gender_classifer.getGender(image_ref, box=bboxes[0])
    
    # 获取模板图像
    template_img, template_face, template_hair = getTemplateImages(template, image_ref_gender)

    # 使用dlib合成方案
    # swapface = FaceSwap(dlib_landmark_predictor)
    # merge = swapface.swap_face(image_ref, template_face)

    # merge = dlibFaceSwap(template_img, image_ref)
    merge = dlibFaceSwap(template_face, image_ref, with_hair=False, template_hair=template_hair)
    
    # cv2.imwrite("./merge.jpg", merge)

    base64_string = base64EncodeImage(merge)

    resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
    resp['error'] = None
    resp['swaped_image']=base64_string
    return json.dumps(resp)

def dlibFaceSwap(template, image_ref, with_hair=True, template_hair=None):
    swapface = FaceSwap(dlib_landmark_predictor)
    if with_hair:
        merge = swapface.swap_face(image_ref, template)
        return merge
    else:
        merge = swapface.swap_face(image_ref, template)
        hair_mask = cv2.threshold(cv2.cvtColor(template_hair, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)
        merge = merge.astype(np.float32)/255.0
        hair_mask =hair_mask[1].astype(np.float32)/255.0
        hair_mask = cv2.merge([hair_mask, hair_mask, hair_mask])
        template_hair = template_hair.astype(np.float32)/255.0
        merge = merge*(1.0 - hair_mask) + template_hair * hair_mask
        merge = (merge*255.0).astype(np.uint8)
        return merge


@main.route("/api/v2/faceswap", methods=['POST'])
@request_required_params(['image_ref', 'template_name'])
@image_required_withkey('image_ref')
@tempate_required_withkey('template_name')
def faceswap_v2(*args, **kwargs):

    image_ref = kwargs['image_ref']
    template = kwargs['template']
    
    resp={}
    resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
    resp['error'] = None
    resp['swaped_image'] = None

    # 检查image_ref中是否有人脸
    faces, bboxes = getFaces(image_ref)
    if len(faces) == 0:
        resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
        resp['error'] = "上传的图像中未检测到人脸"
        resp['swaped_image']=None
        return json.dumps(resp)

    # 检查上传的图像的性别
    image_ref_gender = gender_classifer.getGender(image_ref, box=bboxes[0])
    
    # 获取模板图像
    tempalte_img, template_face, template_hair = getTemplateImages(template, image_ref_gender)

    # 检查image_ref中是否有人脸
    faces, bboxes = getFaces(tempalte_img)
    if len(faces) == 0:
        resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
        resp['error'] = "模板中未检测到人脸"
        resp['swaped_image']=None
        return json.dumps(resp)
    # cv2.imwrite("./tempalte_img.jpg", tempalte_img)
    
    # 获取上传的图像的face landmarks
    image_ref_landmarks = getFaceLandmarks(image_ref)[0]
    
    # 获取模板的face landmarks
    # temlpate_landmarks = getFaceLandmarks(tempalte_img)[0]
    # print(temlpate_landmarks)


    # # 获取提交表单
    # form = request.form
    
    # # 提取表单中的img字段
    # img_ref_base64 = form.get('image_ref')

    # # 读取图片文件
    # img_ref = imageFromBase64Code(img_ref_base64)

    # tempalte_path = "./res/template1/"

    # # 是因为mediapipe获取图像的面部bbox
    # faces, bboxs = getFaces(img_ref)
    # img_ref_gender = gender_classifer.getGender(img_ref, box=bboxs[0])
    # template_name = "template_F.png"
    # if img_ref_gender == 'male':
    #     template_name = "template_M.png"
    # elif img_ref_gender == 'female':
    #     template_name = "template_F.png"
    # else:
    #     pass
    # img_template_path = tempalte_path + template_name

    # img_template = cv2.imread(img_template_path, cv2.IMREAD_COLOR)

    # # 获取landmarks
    # img_tempalte_landmarks = getFaceLandmarks(img_template)[0]
    # img_ref_landmarks = getFaceLandmarks(img_ref)[0]

    # output = faceSwap(img_template, img_ref, img_tempalte_landmarks, img_ref_landmarks)

    
    # base64_string = base64EncodeImage(output)


    return json.dumps({'success':True, 'result': 'base64_string'}), 200, {'ContentType':'application/json'} 


def getTemplateImages(tempalte, gender):
    """获取模板相关的合成文件"""
    tempalte_img_path = tempalte[gender]['img']
    tempalte_face_path = tempalte[gender]['face']
    tempalte_hair_path = tempalte[gender]['hair']

    tempalte_img = cv2.imread(tempalte_img_path, cv2.IMREAD_COLOR)
    tempalte_face = cv2.imread(tempalte_face_path, cv2.IMREAD_COLOR)
    tempalte_hair = cv2.imread(tempalte_hair_path, cv2.IMREAD_COLOR)

    return tempalte_img, tempalte_face, tempalte_hair
