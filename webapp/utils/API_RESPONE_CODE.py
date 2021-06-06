"""
API返回的状态码
"""

class API_RESPONE_CODE:
    """api饭后状态码"""
    API_RESPONE_SUCCESS=200 #请求处理成功

    REQUEST_ARGUMENTS_ERROR=4100 #API参数填写错误
    IMAGE_NOFACE=4101   #图像中未检测到人脸
    FACE_OUTREGION=4102    #人脸区域占比过大
    FACE_LEFT_RIGHT_ROTATION= 4103 # 人脸左右转动角度过大
    FACE_UP_DOWN_ROTATION=4104  # 人脸上下转动角度过大
    FACE_NOLANDMARKS=4105 #无法检测面部的关键特征点


    SERVER_FACEMODULE_ERROR = 5300 # 合成算法错误

