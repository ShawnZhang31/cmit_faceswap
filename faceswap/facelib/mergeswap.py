import cv2
import numpy as np
import argparse

from numpy.core.fromnumeric import product
from faceswap.facelib.facedetect import FaceDetect
import dlib


class FaceMergeSwap(object):
    def __init__(self):
        super().__init__()
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))

        # 用来截取图片.
        # self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS + self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)
        self.ALIGN_POINTS = ( self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS  + self.NOSE_POINTS + self.MOUTH_POINTS)
        # self.ALIGN_POINTS = ( list(range(0, 27)))

        # Points from the second image to overlay on the first. The convex hull of each
        # element will be overlaid.
        # self.OVERLAY_POINTS = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS, self.NOSE_POINTS + self.MOUTH_POINTS,]
        self.OVERLAY_POINTS = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS, self.NOSE_POINTS + self.MOUTH_POINTS,]
        # self.OVERLAY_POINTS = [list(range(0, 27))]
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cmit face swap')
    parser.add_argument('-i', '--image_path', default='./docs/test_imgs/uuxm.jpeg', type=str,
                        help='path to input image')
    parser.add_argument('-t', '--template_path', default='./res/templates/template1/female/female.jpg', type=str,
                        help="path to template image")
    args = parser.parse_args()

    # 输入的图像
    image_ref = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    image_template = cv2.imread(args.template_path, cv2.IMREAD_COLOR)

    dlib_landmark_predictor = dlib.shape_predictor("./res/dlib/shape_predictor_68_face_landmarks.dat")

    # 面部检测器
    faceDetector = FaceDetect(dlib_landmark_predictor)

    # 对用户输入的图像进行检测
    face_bboxes_ref, face_angles_ref = faceDetector.detectFace(image_ref, with_angle=True)
    landmarks_ref = faceDetector.detectFaceLandmarks(image_ref, facebox=face_bboxes_ref[0])

    # 测试用户图像的识别情况
    image_ref_copied = image_ref.copy()
    faceDetector._drawFaceBox(image_ref_copied, face_bboxes_ref[0])
    faceDetector._drawFaceLandmark(image_ref_copied, landmarks_ref)
    cv2.imshow("image_ref", image_ref_copied)

    # 对模板图像进行检测
    face_bboxes_template, face_angles_template = faceDetector.detectFace(image_template, with_angle=True)
    landmarks_template = faceDetector.detectFaceLandmarks(image_template, facebox=face_bboxes_template[0])

    # 测试模板图像的识别情况
    image_template_copied = image_template.copy()
    faceDetector._drawFaceBox(image_template_copied, face_bboxes_template[0])
    faceDetector._drawFaceLandmark(image_template_copied, landmarks_template)
    cv2.imshow("image_template", image_template_copied)
    # cv2.waitKey()
    # print(face_bboxes_template, face_angles_template)

    # 创建合成算法类
    faceMergeSwap = FaceMergeSwap()
    print(faceMergeSwap.OVERLAY_POINTS)
    
    




