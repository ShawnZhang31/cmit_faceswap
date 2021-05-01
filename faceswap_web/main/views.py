# 路由
from flask import request
from . import main
import base64
import cv2
import numpy as np
import json

from faceswap.facelib import DlibToolClass, GenderToolClass
from faceswap.swap import FaceSwap

from faceswap.faceswap_utils.ImageException import ImageError
from faceswap.faceswap_utils.ModelLoadException import ModelLoadError


@main.route("/", methods=['GET', 'POST'])
def index():
    return "Hello Face Swap from CMIT"

# 人脸融合V1
@main.route("/faceswap/v1", methods=['POST'])
def faceswap_v1():
    # 获取提交表单
    form = request.form
    
    # 提取表单中的img字段
    img_base64 = form.get('image')

    #提取数据中有效字段
    data = img_base64.split(',')[1]

    #获取解码后的base64字符
    data = base64.decodebytes(data.encode())

    # 转换为numpy array
    nparr = np.frombuffer(data, np.uint8)

    # 读取图片文件
    img1 = cv2.imdecode(nparr, -1)
    

    gender_prototxt_file_path = "./res/gender/gender_deploy.prototxt"
    gender_net_file_path = "./res/gender/gender_net.caffemodel"

    dlib_shape_model_path = "./res/dlib/shape_predictor_68_face_landmarks.dat"



    tempalte_path = "./res/template1/"

    genderClassfier = GenderToolClass(gender_prototxt_file_path, gender_net_file_path)
    img1_gender = genderClassfier.getGender(img1)
    template_name = "template_F.png"
    if img1_gender == 'Male':
        template_name = "template_M.png"
    elif img1_gender == 'Female':
        template_name = "template_F.png"
    else:
        pass
    img2_path = tempalte_path + template_name

    img2 = cv2.imread(img2_path)


    swapface = FaceSwap(dlib_shape_model_path)
    merge = swapface.swap_face(img1, img2)
    

    base64_string = base64EncodeImage(merge)


    return json.dumps({'success':True, 'result': base64_string}), 200, {'ContentType':'application/json'} 
    # return base64_string

def base64EncodeImage(image, with_base64_header=True, file_ext='jpg'):
    """将图像编码为base64的字符串
        @参数:
            image: 要进行编码的图片，numpy array
            file_ext: 文件后缀名，默认参数为jpg
            with_base64_header: 返回的字符串是否带base64的说明头
        @返回值
            图标进行base64编码之后的字符串
    """
    imageData = cv2.imencode("."+file_ext, image)
    base64_str = imageData[1].tostring()
    base64_str = base64.b64encode(base64_str).decode()
    if with_base64_header:
            base64_str = "data:image/"+file_ext+";base64,"+base64_str
    return base64_str