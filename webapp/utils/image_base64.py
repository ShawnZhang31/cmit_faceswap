import cv2
import base64
import numpy as np
from PIL import Image
import re
from io import BytesIO
from PIL.ExifTags import TAGS


def imageFromBase64Code(img_base64):
    base64_data = re.sub('^data:image/.+;base64,', '', img_base64)

    #获取解码后的base64字符
    try:
        byte_data = base64.b64decode(base64_data)
    except Exception:
        return None, "不是一个有效的图像文件"
    
    image_data = BytesIO(byte_data)

    try:
        image = Image.open(image_data)
    except Exception:
         return None, "不是一个有效的jpg/jpeg/png格式的图像文件"
    # extract EXIF data
    # 获取图像的旋转信息：https://jdhao.github.io/2019/07/31/image_rotation_exif_info/
    
    if image.mode not in ['L', 'RGB', 'RGBA']:
        return None, "仅支持jpg/jpeg/png格式的RGB颜色空间的图像文件"

    exifdata = image.getexif()
    # print(exifdata)
    exif=dict((TAGS[k], v) for k, v in exifdata.items() if k in TAGS)
    if 'Orientation' in exif.keys():
        image_orientation = exif['Orientation']
        if image_orientation == 3: # 旋转180度
            image = image.rotate(180, Image.NEAREST, expand=True)
        if image_orientation == 6: # 选择270度
            image = image.rotate(270, Image.NEAREST, expand=True)
        if image_orientation == 8: # 旋转90度
            image = image.rotate(90, Image.NEAREST, expand=True)
    
    image_array = np.array(image)
    # image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # 
    if len(image_array.shape) == 2:
        # 大通道数据
        image_array = np.array([image_array, image_array, image_array])
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    elif len(image_array.shape) == 3:
        #三通道数据
        heigt, width, channels = image_array.shape
        if channels == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        elif channels == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        else:
            return None, "仅支持jpg/jpeg/png格式的RGB颜色空间的图像文件"
    else:
        return None, "仅支持jpg/jpeg/png格式的RGB颜色空间的图像文件"
    # image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    return image_array, None


    # iterating over all EXIF data fields
    # for tag_id in exifdata:
    # # get the tag name, instead of human unreadable tag id
    #     tag = TAGS.get(tag_id, tag_id)
    #     data = exifdata.get(tag_id)
    #      # decode bytes 
    #     if isinstance(data, bytes):
    #          data = data.decode()
    #     print(f"{tag:25}: {data}")


def imageFromBase64Code_(img_base64):
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
        return None, "不是一个有效的图像文件"

    # 转换为numpy array
    nparr = np.frombuffer(data, np.uint8)

    # 读取图片文件
    image = cv2.imdecode(nparr, -1)
    if image is None:
        return None, "不是一个有效的jpg/jpeg/png格式的图像文件"
    return image, None
    # return image


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