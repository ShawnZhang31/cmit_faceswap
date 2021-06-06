# 路由
import os
import yaml
from flask import request
from . import main
import base64
import cv2
import numpy as np
import json
import math

from webapp import gender_classifer, dlib_landmark_predictor, FaceDetectTool
from webapp.utils import imageFromBase64Code, base64EncodeImage

from faceswap.facelib import DlibToolClass, GenderToolClass, gender
from faceswap.swap import FaceSwap
from faceswap.facelib.mediaface import getFaces, getFaceLandmarks, faceSwap
from faceswap.facelib.mergeswap import FaceMergeSwap

from faceswap.faceswap_utils.ImageException import ImageError
from faceswap.faceswap_utils.ModelLoadException import ModelLoadError

from webapp.utils.decorators.check_image import image_required_withkey
from webapp.utils.decorators.check_params import request_required_params
from webapp.utils.decorators.check_template import getTemplateImgs, tempate_required_withkey
from webapp.utils.API_RESPONE_CODE import API_RESPONE_CODE


GENDER_LIST=['male', 'female']

@main.route("/", methods=['GET', 'POST'])
def index():
    # add = Add(3, 2)
    return "Hello Face Swap from CMIT"
    # return json.dumps({"test":add.sum()}), 200

# 人脸融合V1
@main.route("/api/v1_old/faceswap", methods=['POST'])
@request_required_params(['image_ref', 'template_name'])
@image_required_withkey('image_ref')
@tempate_required_withkey('template_name')
def faceswap_v1_old(*args, **kwargs):
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
    merge, message = dlibFaceSwap(template_face, image_ref, with_hair=False, template_hair=template_hair)
    
    # cv2.imwrite("./merge.jpg", merge)
    if merge is None:
        resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
        resp['error'] = message
        resp['swaped_image']=None
        return json.dumps(resp)

    base64_string = base64EncodeImage(merge)

    resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
    resp['error'] = None
    resp['swaped_image']=base64_string
    return json.dumps(resp)

def dlibFaceSwap(template, image_ref, with_hair=True, template_hair=None):
    swapface = FaceSwap(dlib_landmark_predictor)
    if with_hair:
        try:
            merge = swapface.swap_face(image_ref, template)
        except ImageError as e:
            return None, e.message
        return merge, None
    else:
        try:
            merge = swapface.swap_face(image_ref, template)
        except ImageError as e:
            return None, e.message
        merge = swapface.swap_face(image_ref, template)
        hair_mask = cv2.threshold(cv2.cvtColor(template_hair, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)
        merge = merge.astype(np.float32)/255.0
        hair_mask =hair_mask[1].astype(np.float32)/255.0
        hair_mask = cv2.merge([hair_mask, hair_mask, hair_mask])
        template_hair = template_hair.astype(np.float32)/255.0
        merge = merge*(1.0 - hair_mask) + template_hair * hair_mask
        merge = (merge*255.0).astype(np.uint8)
        return merge, None


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


@main.route("/api/v1/faceswap", methods=['POST'])
@request_required_params(['image_ref', 'template_name'])
@image_required_withkey('image_ref')
@tempate_required_withkey('template_name')
def faceswap_v1(*args, **kwargs):
    #检查提交参数
    image_ref = kwargs['image_ref']
    template = kwargs['template']

    # 返回体
    resp={}
    resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
    resp['error'] = None
    resp['swaped_image'] = None

    # # 检查用户提交的图片中是否有人脸，及人脸是否摆正
    image_ref_face_deteced_result = json.loads(face_swap_detect(image_ref))
    if image_ref_face_deteced_result['code'] != API_RESPONE_CODE.API_RESPONE_SUCCESS:
        # 人脸检测不合格
        resp['code'] = image_ref_face_deteced_result['code']
        resp['error'] = image_ref_face_deteced_result['error']
        resp['swaped_image'] = None
        return json.dumps(resp)

    # 检查用户是否提交gender
    image_ref_gender = None
    if 'gender' not in  request.form:
        image_ref_gender = None
    else:
        image_ref_gender = request.form.get('gender')
        if image_ref_gender not in GENDER_LIST:
            image_ref_gender = None
        else:
            pass
    # 如果image_ref_gender为None则要检测性别
    if image_ref_gender is None:
        box_dict = image_ref_face_deteced_result['face']['box']
        face_box = [box_dict['left'], 
                    box_dict['top'], 
                    box_dict['left']+box_dict['width'],
                    box_dict['top']+box_dict['height']]
        image_ref_gender = gender_classifer.getGender(image_ref, face_box)

    # 检查swaped_value参数
    swaped_value = None
    if 'swaped_value' not in  request.form:
        swaped_value = None
    else:
        swaped_value = request.form.get('swaped_value')
        try:
            swaped_value = int(swaped_value)
        except Exception:
            swaped_value = 39

    if swaped_value is None:    # 默认是21
        swaped_value = 39
    
    if swaped_value%2 == 0:     # 必须是奇数
        swaped_value += 1

    if swaped_value < 31:       # 不能小于11
        swaped_value = 31
    if swaped_value > 61:       # 不能大于31
        swaped_value = 61 

    # 检测文件后缀名
    swaped_image_ext = None
    if 'swaped_image_ext' not in  request.form:
        swaped_image_ext = None
    else:
        swaped_image_ext = request.form.get('swaped_image_ext')
        if swaped_image_ext not in ['png', 'jpg']:
            swaped_image_ext = 'png'
    if swaped_image_ext is None:
        swaped_image_ext = 'png'


    # 获取模板图像
    template_img, template_face, template_hair_mask = getTemplateImages(template, image_ref_gender)

    # 开始进行合成前的检测
    # 用户提交的图像的检测
    box_dict = image_ref_face_deteced_result['face']['box']
    face_bboxes_ref = [box_dict['left'], 
                        box_dict['top'], 
                        box_dict['width'], 
                        box_dict['height']]
    landmarks_ref = FaceDetectTool.detectFaceLandmarks(image_ref, facebox=face_bboxes_ref)
    # 测试用户提交的图片的识别情况
    image_ref_copied = image_ref.copy()
    FaceDetectTool._drawFaceBox(image_ref_copied, face_bboxes_ref)
    FaceDetectTool._drawFaceLandmark(image_ref_copied, landmarks_ref)
    cv2.imwrite("image_ref_detected.jpg", image_ref_copied)
    
    # 模板的检测
    # 对模板图像进行检测
    face_bboxes_template, face_angles_template = FaceDetectTool.detectFace(template_face, with_angle=False)
    landmarks_template = FaceDetectTool.detectFaceLandmarks(template_face, facebox=face_bboxes_template[0])

    # 测试模板图像的识别情况
    image_template_copied = template_img.copy()
    FaceDetectTool._drawFaceBox(image_template_copied, face_bboxes_template[0])
    FaceDetectTool._drawFaceLandmark(image_template_copied, landmarks_template)
    cv2.imwrite("image_template_detected.jpg", image_template_copied)

    # 开始进行合成
    # 创建合成算法类
    faceMergeSwap = FaceMergeSwap(FEATHER_AMOUNT=swaped_value)
    image_swaped = faceMergeSwap.swap(template_face, image_ref, landmarks_template, landmarks_ref)

    # 与头发融合
    template_img = template_img.astype(np.float64)
    image_swaped = image_swaped.astype(np.float64)
    template_hair_mask = template_hair_mask.astype(np.float64)/255.0
    
    merge = template_img * template_hair_mask + image_swaped * (1.0 - template_hair_mask)
    merge = merge.astype(np.uint8)
    
    cv2.imwrite("./merge.jpg", merge)

    base64_string = base64EncodeImage(merge, file_ext=swaped_image_ext)

    resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
    resp['error'] = None
    resp['swaped_image']=base64_string
    return json.dumps(resp)

@main.route("/api/v1/face_detect", methods=['POST'])
@request_required_params(['image_ref'])
@image_required_withkey('image_ref')
def faces_detect(*args, **kwargs):
    #检查提交参数
    image_ref = kwargs['image_ref']
    return face_swap_detect(image_ref)

def face_swap_detect(image_ref):
    resp = {}
    resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
    resp['error'] = None
    resp['face']=None

    face_bboxes, face_eulerangles = FaceDetectTool.detectFace(image_ref, with_angle=True)

    if len(face_bboxes) <=0: # 未检测到人脸
        resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
        resp['error'] = "图像中未检测的人脸"
        resp['face']=None
        return json.dumps(resp)
    else:
        face_box = face_bboxes[0]

        if face_box[0] <=0 or face_box[1] <=0:
            resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
            resp['error'] = "图像虽然检测到人脸, 但面部部分特征点超出了图像区域，不符合合成要求"
            resp['face']=None
            return json.dumps(resp)

        face_euler_angle = face_eulerangles[0]
        if face_euler_angle is None:
            resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
            resp['error'] = "图像虽然检测到人脸，但无法计算面部欧拉角,不符合合成要求"
            resp['face']=None
            return json.dumps(resp)
        else:

            face_orientation_check, reason = checkFaceOrientation(face_euler_angle)
            if face_orientation_check == False:
                resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
                resp['error'] = "图像虽然检测到人脸, 但面部旋转角度超出设定阈值, "+reason
                resp['face']=None
                return json.dumps(resp)
            else:
                face_info=dict()
                face_info['box'] ={"left":face_box[0],
                                    "top":face_box[1],
                                    "width":face_box[2],
                                    "height":face_box[3]}
                face_info['euler_angle']=face_euler_angle
                resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
                resp['error'] = None
                resp['face']=face_info
                return json.dumps(resp)

    return json.dumps(resp)


def checkFaceOrientation(euler_angles):
    """检查人脸的朝向是否合适"""
    face_thres_path = os.getenv('FACE_EULER_ANGLES_THRESH') or './res/face/face.yaml'
    with open(face_thres_path, 'r', encoding='utf-8') as f:
        face_thres = yaml.load(f, Loader=yaml.SafeLoader)['angle']
        f.close()
    status = True
    reason = None
    if math.fabs(euler_angles['pitch']) >= face_thres['pitch']:
        status = False
        reason = "请不要低头或抬头, 请正对摄像头"
    if math.fabs(euler_angles['yaw']) >= face_thres['yaw']:
        status = False
        reason = "请不要左右转头, 请正对摄像头"
    return status, reason