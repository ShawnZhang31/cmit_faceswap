from math import degrees
import cv2
import numpy as np
import argparse

import faceswap.facelib.faceBlendCommon as fbc
from faceswap.facelib.facedetect import FaceDetect
import dlib


class FaceMergeSwap(object):
    def __init__(self, FEATHER_AMOUNT=31):
        super().__init__()
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))
        self.FACE_COUNTOUR=[[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]]
        self.FEATHER_AMOUNT = FEATHER_AMOUNT# 39 will be better

        # 用来截取图片.
        # self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS + self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)
        self.ALIGN_POINTS = ( self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS  + self.NOSE_POINTS + self.MOUTH_POINTS)
        # self.ALIGN_POINTS = ( list(range(0, 27)))
        # self.OVERLAY_POINTS = (list([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]))

        # Points from the second image to overlay on the first. The convex hull of each
        # element will be overlaid.
        # self.OVERLAY_POINTS = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS, self.NOSE_POINTS + self.MOUTH_POINTS,]
        self.OVERLAY_POINTS = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS, self.NOSE_POINTS + self.MOUTH_POINTS,]
        # self.OVERLAY_POINTS = [list(range(0, 27))]
        # self.OVERLAY_POINTS = (list([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17]))
    
    def draw_convex_hull(self, im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def cal_convex_hull(self, points):
        return cv2.convexHull(points, returnPoints=False)
    

    def get_mask_center_point(self,image_mask):
        """
        获取掩模的中心点坐标
        :param image_mask: 掩模图片
        :return: 掩模中心
        """
        image_mask_index = np.argwhere(image_mask > 0)
        miny, minx = np.min(image_mask_index, axis=0)
        maxy, maxx = np.max(image_mask_index, axis=0)
        center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
        return center_point
    
    def getDelanuWrapedImageAndMask(self, image_template, image_ref, image_tempalted_landmarks, image_ref_landmarks):
        """获取使用Delanunary细分变形的图像和遮罩"""
        image_tempalted_Warped = np.copy(image_template)
        
        # 计算hull
        hull_tempalteds = []
        hull_refs = []

        
        # warped_mask = np.zeros(image_template.shape[:2], dtype=np.float64)
        # for group in self.OVERLAY_POINTS:
        # print(np.array(self.FACE_COUNTOUR))
        for group in np.array(self.FACE_COUNTOUR):
        # for group in np.array(self.OVERLAY_POINTS):
            hullIndex = self.cal_convex_hull(np.array(image_tempalted_landmarks))
            # hullIndex = self.cal_convex_hull(np.array(image_tempalted_landmarks)[group])
            # self.draw_convex_hull(warped_mask, np.array(image_tempalted_landmarks)[group], 1)
            hull_t = []
            hull_r = []
            for idx in range(0, len(hullIndex)):
                hull_t.append(image_tempalted_landmarks[hullIndex[idx][0]])
                hull_r.append(image_ref_landmarks[hullIndex[idx][0]])
            hull_tempalteds.append(hull_t)
            hull_refs.append(hull_r)

        sizeImage_ref = image_ref.shape
        rect = (0, 0, sizeImage_ref[1], sizeImage_ref[0])

        # 针对不同的合成区域进行三角细分
        for h_idx , hull_ref in enumerate(hull_refs):
            # 三角细分
            dt = fbc.calculateDelaunayTriangles(rect, hull_ref)
            # print(dt)

            # 对细分三角形进行变换
            for idx in range(0, len(dt)):
                t1 = []
                t2 = []
                
                # 获取模板图像和参考图像上对应的三角形
                for j in range(0, 3):
                    t1.append(hull_tempalteds[h_idx][dt[idx][j]])
                    t2.append(hull_ref[dt[idx][j]])
                
                fbc.warpTriangle(image_tempalted_Warped, image_ref, t1, t2)
                # cv2.imshow("image_tempalted_Warped", image_tempalted_Warped)
                # cv2.waitKey(300)
        # cv2.imshow("image_tempalted_Warped", image_tempalted_Warped)
        # cv2.imshow("warped_mask", warped_mask)
        # cv2.waitKey()

        #计算overlayer区域的mask
        overlayed_mask = np.zeros(image_template.shape[:2], dtype=np.float64)
        for group in np.array(self.OVERLAY_POINTS, dtype=object):
            _ = self.draw_convex_hull(overlayed_mask, np.array(image_tempalted_landmarks)[group], 1)

        

        # print(overlayed_mask.shape)
        # 将overlayer转化为3通道数据
        overlayed_mask = np.array([overlayed_mask, overlayed_mask, overlayed_mask]).transpose((1, 2, 0))

        # 进行边缘虚化处理
        overlayed_mask = (cv2.GaussianBlur(overlayed_mask, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        overlayed_mask = cv2.GaussianBlur(overlayed_mask, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)
        overlayed_mask = np.uint8(overlayed_mask*255)

        return image_tempalted_Warped, overlayed_mask

    def transformation_from_points(self, points1, points2):
        """获取从points1 -> points2的affine transformation"""
        points1 = np.matrix([[p[0], p[1]] for p in points1])
        points2 = np.matrix([[p[0], p[1]] for p in points2])
        points1 =  points1.astype(np.float64)
        points2 =  points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2
        
        # print(type(points1.T))
        U, S, Vt = np.linalg.svd(points1.T * points2)
        

        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R, 
                                    c2.T - (s2 / s1) * R * c1.T)), 
                                    np.matrix([0., 0., 1.])])

    def warp_im(self, im, M, dshape):
        """对图像进行仿射变换"""
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
        return output_im


    def getAffineWrapedImageAndMask(self, image_template, image_ref, image_tempalted_landmarks, image_ref_landmarks):
        """获取仿射变化的图像和遮罩"""

        M = self.transformation_from_points(np.array(image_tempalted_landmarks)[self.ALIGN_POINTS], np.array(image_ref_landmarks)[self.ALIGN_POINTS])

        # points1 = np.array(image_tempalted_landmarks, order='C')[self.ALIGN_POINTS]
        # points2 = np.array(image_ref_landmarks, order='C')[self.ALIGN_POINTS]
        # points1 =  points1.astype(np.float)
        # points2 =  points2.astype(np.float)
        # # print(points1)
        # M, _= cv2.estimateAffine2D(points1, points2, cv2.LMEDS)
        # M, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

         #计算overlayer区域的mask
        overlayed_mask = np.zeros(image_template.shape[:2], dtype=np.float64)
        for group in np.array(self.OVERLAY_POINTS, dtype=object):
            _ = self.draw_convex_hull(overlayed_mask, np.array(image_tempalted_landmarks)[group], 1)
        
        # print(overlayed_mask.shape)
        # 将overlayer转化为3通道数据
        overlayed_mask = np.array([overlayed_mask, overlayed_mask, overlayed_mask]).transpose((1, 2, 0))

        # 进行边缘虚化处理
        overlayed_mask = (cv2.GaussianBlur(overlayed_mask, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0) > 0) * 1.0
        overlayed_mask = cv2.GaussianBlur(overlayed_mask, (self.FEATHER_AMOUNT, self.FEATHER_AMOUNT), 0)
        overlayed_mask = np.uint8(overlayed_mask*255)

        image_tempalted_Warped = np.copy(image_ref)
        image_tempalted_Warped = self.warp_im(image_tempalted_Warped, M, image_template.shape)
        # img_size = image_template.shape
        # image_tempalted_Warped = cv2.warpPerspective(image_tempalted_Warped, M, (img_size[1], img_size[0]))

        return image_tempalted_Warped, overlayed_mask



    def swap(self, image_template, image_ref, image_tempalted_landmarks, image_ref_landmarks, method="delanu"):
        """根据指定的轮廓进行面部融合"""
        if method == "delanu":
            image_tempalted_Warped, overlayed_mask = self.getDelanuWrapedImageAndMask(image_template, 
                                                                                        image_ref, 
                                                                                        image_tempalted_landmarks,
                                                                                        image_ref_landmarks)
        else:
            image_tempalted_Warped, overlayed_mask = self.getAffineWrapedImageAndMask(image_template, 
                                                                                        image_ref, 
                                                                                        image_tempalted_landmarks,
                                                                                        image_ref_landmarks)

        mask_center = self.get_mask_center_point(cv2.cvtColor(overlayed_mask, cv2.COLOR_BGR2GRAY))
        # cv2.circle(overlayed_mask, mask_center, 2, (0, 0, 255), thickness=2)
        # cv2.imshow("overlayed_mask", overlayed_mask)
        # cv2.imshow("image_tempalted_Warped", image_tempalted_Warped)

        # 计算融合区域的重心，使用单通道图像计算
        

        # 开始合成图像
        # cv::NORMAL_CLONE, cv::MIXED_CLONE or cv::MONOCHROME_TRANSFER
        # re = cv2.boundingRect(np.array(hull_tempalteds[0], np.float32))
        
        image_swaped = cv2.seamlessClone(image_tempalted_Warped, 
                                        image_template, 
                                        overlayed_mask, 
                                        mask_center, cv2.NORMAL_CLONE)
        # cv2.imshow("image_swaped", image_swaped)
        # cv2.waitKey()
        return image_swaped

    def imageLUT(self, image):
        originalValue = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 255])
        adjustCurve = np.array([0, 25, 50, 75, 100, 120, 140, 150, 170, 190, 210, 255])
        # adjustCurve = np.array([0, 50, 100, 150, 175, 210])
        # adjustCurve = np.array([0, 50, 100, 140, 150, 230])
        # adjustCurve = np.array([0, 20,  40,  75, 150, 175])
        

        # Create a LookUp Table
        fullRange = np.arange(0,256)
        adjustLUT = np.interp(fullRange, originalValue, adjustCurve)
        
        imLUTed = image.copy()
        imLUTed = cv2.cvtColor(imLUTed, cv2.COLOR_BGR2Lab)

        # l_channle = imLUTed[:,:,0].copy()

        imLUTed[:,:,0] = cv2.LUT(imLUTed[:,:,0], adjustLUT)

        # l_channle_lut = imLUTed[:,:,0].copy()

        # cv2.imshow("L", np.hstack([l_channle, l_channle_lut]))

        # imLUTed[:,:,1] = cv2.LUT(imLUTed[:,:,1], adjustLUT)
        # imLUTed[:,:,2] = cv2.LUT(imLUTed[:,:,2], adjustLUT)
        imLUTed = cv2.cvtColor(imLUTed, cv2.COLOR_Lab2BGR)
        
        return imLUTed


    def imageCLAHE(self, image):
        """对图像上的高光进行均衡化处理"""
        imhsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        imhsvCLAHE = imhsv.copy()

        # 只在V通道上进行均衡化处理
        imhsv[:,:,2] = cv2.equalizeHist(imhsv[:,:,2])

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        imhsvCLAHE[:,:,2] = clahe.apply(imhsvCLAHE[:,:,2])

        imEq = cv2.cvtColor(imhsv, cv2.COLOR_HSV2BGR)
        imEqCLAHE = cv2.cvtColor(imhsvCLAHE, cv2.COLOR_HSV2BGR)

        return imEq, imEqCLAHE



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
    print(face_bboxes_ref, face_angles_ref)

    # 测试用户图像的识别情况
    image_ref_copied = image_ref.copy()
    faceDetector._drawFaceBox(image_ref_copied, face_bboxes_ref[0])
    faceDetector._drawFaceLandmark(image_ref_copied, landmarks_ref)
    cv2.imshow("image_ref", image_ref_copied)

    # 对模板图像进行检测
    face_bboxes_template, face_angles_template = faceDetector.detectFace(image_template, with_angle=True)
    landmarks_template = faceDetector.detectFaceLandmarks(image_template, facebox=face_bboxes_template[0])
    print(face_bboxes_template, face_angles_template)

    # 测试模板图像的识别情况
    image_template_copied = image_template.copy()
    faceDetector._drawFaceBox(image_template_copied, face_bboxes_template[0])
    faceDetector._drawFaceLandmark(image_template_copied, landmarks_template)
    cv2.imshow("image_template", image_template_copied)

    cv2.waitKey()
    
    # print(face_bboxes_template, face_angles_template)

    # 创建合成算法类
    faceMergeSwap = FaceMergeSwap()
    # print(faceMergeSwap.OVERLAY_POINTS)
    # image_template_LUTED = faceMergeSwap.imageLUT(image_template)
    # cv2.waitKey()
    image_ref_LUTED = faceMergeSwap.imageLUT(image_ref)
    # imeq, imclahe=faceMergeSwap.imageCLAHE(image_ref)

    image_swaped = faceMergeSwap.swap(image_template, image_ref, landmarks_template, landmarks_ref)
    image_swaped_luted = faceMergeSwap.swap(image_template, image_ref_LUTED, landmarks_template, landmarks_ref)
    
    cv2.imshow("image_swaped", np.hstack([image_swaped, image_swaped_luted]))

    # image_swaped_eq = faceMergeSwap.swap(image_template, imeq, landmarks_template, landmarks_ref)
    # image_swaped_clahe = faceMergeSwap.swap(image_template, imclahe, landmarks_template, landmarks_ref)
    # cv2.imshow("image_swaped_eq", image_swaped_eq)
    # cv2.imshow("image_swaped_clahe", image_swaped_clahe)
    cv2.waitKey()
    




