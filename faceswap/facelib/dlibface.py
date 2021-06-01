import numpy as np
import dlib
import cv2
import os
import sys

# from faceswap.faceswap_utils.ModelLoadException import ModelLoadError
from faceswap.faceswap_utils.ModelLoadException import ModelLoadError
# from faceswap.faceswap_utils.ImageException import ImageError
from faceswap.faceswap_utils.ImageException import ImageError

class DlibToolClass():
    
    landmark_predictor=None
    detector = dlib.get_frontal_face_detector()

    # def __init__(self):
    #     pass

    def __init__(self, dlib_shape_predictor):
        self.landmark_predictor = dlib_shape_predictor
    
    def imgvalidate(self,img):
        if img is None:
            raise ImageError(message='图片为空')
        if img.dtype is 'uint8':
            raise ImageError(message='图片必须未uint8')


    def get_landmarks(self,img):
        faces=self.detector(img,1)
        if (len(faces) > 0):
            landmarksPoints=np.zeros((len(faces)*68,2))
            for k,d in enumerate(faces):    
                shape = self.landmark_predictor(img,d)
                for i in range(68):
                    landmarksPoints[i]=[(int)(shape.part(i).x), (int)(shape.part(i).y)]
            return landmarksPoints
        raise ImageError(message='未检测到人脸')


    # 检测人脸
    def detect_face(self,img):
        self.imgvalidate(img)
        faces=self.detector(img,1)
        result=[]
        if len(faces)<1:
            raise ImageError("no face detected")
        for i, d in enumerate(faces):
            x1, y1, x2, y2 = d.left(), d.top(), d.right() , d.bottom()
            result.append({'rectangle': {'x1': x1, 'y1': y1, 'x2': x2,'y2':y2}})
        return result

        # 检测人脸关键点
    def detect_landmark(self,img):
        self.imgvalidate(img)
        faces=self.detector(img,1)
        result=[]
        if (len(faces) > 0):
            for k,d in enumerate(faces):
               
                shape = self.landmark_predictor(img,d)
                landmarksPoints=[]
                for i in range(68):
                    landmarksPoints.append({"x":(int)(shape.part(i).x), 'y':(int)(shape.part(i).y)})
                x1, y1, x2, y2 = d.left(), d.top(), d.right() , d.bottom()
                result.append({'rectangle': {'x1': x1, 'y1': y1, 'x2': x2,'y2':y2},'landmarks':landmarksPoints})  
        else:
            raise ImageError("no face detected")  
        return result