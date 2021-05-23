"""
检验用于提交的图片数据是否符合遥要求
"""
import json
from functools import wraps
from flask import request, current_app

from webapp.utils.API_RESPONE_CODE import API_RESPONE_CODE
from webapp.utils.image_base64 import imageFromBase64Code
import base64
import cv2
import numpy as np

def image_required_withkey(imageKey='image_ref'):
    """检查用于提交的图像是否合乎要求"""
    def image_required_withkey_decorator(f):
        @wraps(f)
        def decorate_function(*args, **kwargs):
            resp={}
            if imageKey not in request.form:
                resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
                resp['error'] = imageKey+"是必填参数"
                resp['swaped_image']=None
                return json.dumps(resp)
            else:
                image_base64code = request.form.get(imageKey)
                image = imageFromBase64Code(image_base64code)
                if image is None:
                    resp['code'] = API_RESPONE_CODE.REQUEST_ARGUMENTS_ERROR
                    resp['error'] = imageKey+"不是一个有效的图像base64编码"
                    resp['swaped_image']=None
                    return json.dumps(resp)
                else:
                    # 检查图像的通道情况
                    img_height, img_width, img_channel = image.shape
                    print(image.shape)
                    if img_channel == 3:
                        pass
                    elif img_channel == 4:
                        channels = []
                        B,G,R,A = cv2.split(image)
                        image = cv2.merge([B, G, R])
                    elif img_channel == 1:
                        image = cv2.merge([image, image, image])
                    else:
                        resp['code'] = API_RESPONE_CODE.API_RESPONE_SUCCESS
                        resp['error'] = "仅支持通道为1, 3, 4的图像合成"
                        resp['swaped_image']=None
                        return json.dumps(resp)
                
                kwargs[imageKey]=image
                return f(*args, **kwargs)
        return decorate_function
    return image_required_withkey_decorator
