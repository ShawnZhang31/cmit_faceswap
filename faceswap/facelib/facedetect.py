import mediapipe as mp
import dlib
import cv2
import numpy as np
import argparse

from numpy.core.fromnumeric import resize, shape
from numpy.lib.function_base import angle, select
from numpy.lib.twodim_base import eye

class FaceDetect(object):
    """实现mediapipe和dlib人脸检测能力"""
    def __init__(self, dlib_shape_predictor) -> None:
        super().__init__()
        self.dlib_face_detector = dlib.get_frontal_face_detector()
        self.dlib_face_predictor = dlib_shape_predictor
        self.mp_face_detector = mp.solutions.face_detection
        self.mp_face_predictor = mp.solutions.face_mesh

    def detectFace(self, image, max_num_face=1, detect_image_width=320, engine='dlib', with_angle=False):
        # 获取图像的size
        img_height, img_width, _ = image.shape
        IMAGE_SCALE_FACTOR = 1.0
        face_bboxes=[]
        face_angles=[]
        image_scaled = image.copy()
        if img_width > detect_image_width:
            IMAGE_SCALE_FACTOR = float(detect_image_width)/float(img_width)
            image_scaled = cv2.resize(image, None, fx=IMAGE_SCALE_FACTOR, fy=IMAGE_SCALE_FACTOR)
        
        if engine == 'dlib':
            bboxes, face_angles = self.__detectDlibFace(image_scaled, with_angle)
            for box in bboxes:
                face_bboxes.append((int(box[0]/IMAGE_SCALE_FACTOR), 
                                    int(box[1]/IMAGE_SCALE_FACTOR), 
                                    int(box[2]/IMAGE_SCALE_FACTOR), 
                                    int(box[3]/IMAGE_SCALE_FACTOR)))
        # print(face_bboxes, face_angles)
        # for box in face_bboxes:
        #     self._drawFaceBox(image, box)
        # cv2.imshow("image", image)
        # cv2.waitKey()
        return face_bboxes, face_angles
    
    def detectFaceLandmarks(self, image, facebox=None, engine='dlib'):
        """检测面部的关键特征点"""
        landmarks = []
        if engine == 'dlib':
            landmarks = self.__dlibDetectFaceLandmarks(image, facebox)
        # self._drawFaceLandmark(image, landmarks)
        # cv2.imshow("landmarks", image)
        # cv2.waitKey()
        return landmarks
        
    def __dlibDetectFaceLandmarks(self, image, facebox):
        face_rect = dlib.rectangle(facebox[0],
                                   facebox[1],
                                   facebox[0]+facebox[2],
                                   facebox[1]+facebox[3])
        shape = self.dlib_face_predictor(image, face_rect)
        return self.__dlib_shape_to_array(shape)
        

    def __detectDlibFace(self, image, with_angle=False, max_num_face=1):
        """使用dlib来检测面部"""
        bbox=[] 
        angles=[]
        rects = self.dlib_face_detector(image, 1)
        if len(rects) <= 0:
            return bbox, angles
        else:
            face_rects = []
            for idx in range(0, min(max_num_face, len(rects))):
                face_rects.append(rects[idx])
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for rect in face_rects:
            bbox.append(self.__dlib_rect_to_bb(rect))
            if with_angle:
                # 计算面部的朝向
                shape = self.dlib_face_predictor(image_gray, rect)
                # 计算水平旋转方向
                left_eye_inner = np.array([shape.part(39).x, shape.part(39).y])
                right_eye_inner = np.array([shape.part(42).x, shape.part(42).y])
                eye_vec = right_eye_inner - left_eye_inner
                ref_vec = np.array([image.shape[1], 0])
                cosangle = eye_vec.dot(ref_vec)/(np.linalg.norm(eye_vec) * np.linalg.norm(ref_vec))
                angle_h = np.abs((np.arccos(cosangle)*180.0)/np.pi)
                # 计算数值旋转方向
                nose_bottom = np.array([shape.part(27).x, shape.part(27).y])
                nose_top = np.array([shape.part(30).x, shape.part(30).y])
                nose_vec = nose_top - nose_bottom
                ref_vec = np.array([0, image.shape[0]])
                cosangle = nose_vec.dot(ref_vec)/(np.linalg.norm(nose_vec) * np.linalg.norm(ref_vec))
                angle_v = np.abs((np.arccos(cosangle)*180.0)/np.pi)
                angles.append((angle_h, angle_v))
        return bbox, angles


    
    def __dlib_rect_to_bb(self, rect):
        """将dlib的rect转化为Face Bounding Box"""
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)
    
    def __dlib_shape_to_array(self, shape):
        """将dlib的shape转化为数组"""
        coords = []
        for i in range(0, 68):
            coords.append((int(shape.part(i).x), int(shape.part(i).y)))
        return coords

    def _drawFaceBox(self, image, box):
        """绘制脸部的Bounding Box"""
        # left_top = 
        cv2.rectangle(image, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), thickness=2)

    def _drawFaceLandmark(self, image, landmarks):
        """绘制脸部的Landmarks"""
        for point in landmarks:
            cv2.circle(image, point, 1, (0, 255, 0), thickness=1)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cmit face detect')
    parser.add_argument('-i', '--image_path', default='./docs/test_imgs/dalk.png', type=str,
                        help='path to input image')
    args = parser.parse_args()

    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)

    dlib_landmark_predictor = dlib.shape_predictor("./res/dlib/shape_predictor_68_face_landmarks.dat")

    faceDetector = FaceDetect(dlib_landmark_predictor)
    face_bboxes, face_angles = faceDetector.detectFace(image, with_angle=True)
    landmarks = faceDetector.detectFaceLandmarks(image, facebox=face_bboxes[0])
