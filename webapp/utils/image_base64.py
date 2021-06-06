import cv2
import base64
import numpy as np

def imageFromBase64Code(img_base64):
    """
    将base64编码的图像还原为OpenCV的图像
    """
    #提取数据中有效字段
    if ',' in img_base64:
        data = img_base64.split(',')[1]
    else:
        data = img_base64

    #获取解码后的base64字符
    try:
        data = base64.decodebytes(data.encode())
    except Exception:
        return None

    # 转换为numpy array
    nparr = np.frombuffer(data, np.uint8)

    # 读取图片文件
    image = cv2.imdecode(nparr, -1)
    return image


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