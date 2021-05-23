import argparse
import cv2
import numpy as np
import dlib
import argparse
# from facedlib.dlibface import DlibToolClass
from faceswap.facelib import DlibToolClass, GenderToolClass

from faceswap.faceswap_utils.ImageException import ImageError
from faceswap.faceswap_utils.ModelLoadException import ModelLoadError

class FaceSwap:
    
    def __init__(self, dlib_shape_predictor):
        self.dlibTool = DlibToolClass(dlib_shape_predictor)

    def get_image_size(self,image):
        """
        获取图片大小（高度,宽度）
        :param image: image
        :return: （高度,宽度）
        """
        image_size = (image.shape[0], image.shape[1])
        return image_size
    
    def get_face_mask(self,image_size, face_landmarks):
        """
        获取人脸掩模
        :param image_size: 图片大小
        :param face_landmarks: 68个特征点
        :return: image_mask, 掩模图片
        """
        mask = np.zeros(image_size, dtype=np.uint8)
        points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
        cv2.fillPoly(img=mask, pts=[points], color=255)

        # mask = np.zeros(image_size, dtype=np.uint8)
        # points = cv2.convexHull(face_landmarks)  # 凸包
        # cv2.fillConvexPoly(mask, points, color=255)
        return mask

    def get_affine_image(self,image1, image2, face_landmarks1, face_landmarks2):
        """
        获取图片1仿射变换后的图片
        :param image1: 图片1, 要进行仿射变换的图片
        :param image2: 图片2, 只要用来获取图片大小，生成与之大小相同的仿射变换图片
        :param face_landmarks1: 图片1的人脸特征点
        :param face_landmarks2: 图片2的人脸特征点
        :return: 仿射变换后的图片
        """
        three_points_index = [18, 8, 25]
        M = cv2.getAffineTransform(face_landmarks1[three_points_index].astype(np.float32),
                                face_landmarks2[three_points_index].astype(np.float32))
        dsize = (image2.shape[1], image2.shape[0])
        affine_image = cv2.warpAffine(image1, M, dsize)
        return affine_image.astype(np.uint8)

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
    
    def get_mask_union(self,mask1, mask2):
        """
        获取两个掩模掩盖部分的并集
        :param mask1: mask_image, 掩模1
        :param mask2: mask_image, 掩模2
        :return: 两个掩模掩盖部分的并集
        """
        mask = np.min([mask1, mask2], axis=0)  # 掩盖部分并集
        mask = ((cv2.blur(mask, (3, 3)) == 255) * 255).astype(np.uint8)  # 缩小掩模大小
        mask = cv2.blur(mask, (5, 5)).astype(np.uint8)  # 模糊掩模
        return mask


    def imgvalidate(self,img):
        if img is None:
            raise ImageError(message='图片为空')
        if img.dtype is 'uint8':
            raise ImageError(message='图片必须未uint8')

    def swap_face(self,img1,img2):
        self.imgvalidate(img1)
        self.imgvalidate(img2)
        
        points1=self.dlibTool.get_landmarks(img1)
        points2=self.dlibTool.get_landmarks(img2)
        if(points1 is None or points2 is None):
            raise ImageError("no face detected")
        img1_size=self.get_image_size(img1)
        img1_mask = self.get_face_mask(img1_size, np.array(points1,dtype=np.int))  # 脸图人脸掩模
        # cv2.imshow("img1_mask", img1_mask)
        # cv2.waitKey(0)
        img2_size = self.get_image_size(img2)  # 摄像头图片大小
        img2_mask = self.get_face_mask(img2_size, np.array(points2,dtype=np.int))  # 摄像头图片人脸掩模
        # cv2.imshow("img2_mask", img2_mask)
        # cv2.waitKey(0)
        affine_img1 = self.get_affine_image(img1, img2, points1, points2)  # im1（脸图）仿射变换后的图片
        affine_img1_mask = self.get_affine_image(img1_mask, img2, points1, points2)  # im1（脸图）仿射变换后的图片的人脸掩模

        union_mask = self.get_mask_union(img2_mask, affine_img1_mask)  # 掩模合并
        # cv2.imshow("union_mask", union_mask)
        # cv2.waitKey(0)
        point = self.get_mask_center_point(affine_img1_mask)  # im1（脸图）仿射变换后的图片的人脸掩模的中心点
        seamless_im = cv2.seamlessClone(affine_img1, img2, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)  # 进行泊松融合
        return seamless_im

if __name__ == "__main__":
    
    gender_prototxt_file_path = "./res/gender/gender_deploy.prototxt"
    gender_net_file_path = "./res/gender/gender_net.caffemodel"

    dlib_shape_model_path = "./res/dlib/shape_predictor_68_face_landmarks.dat"

    parser = argparse.ArgumentParser(description='Meadiapip face swap')
    parser.add_argument('-i', '--image_path', default='./res/template1/template_F.png', type=str,
                        help='path to input image')
    parser.add_argument('-r', '--ref_path', default='./res/img1.jpeg', type=str, 
                        help='path to reference image(texture ref)')
    args = parser.parse_args()
    image = cv2.imread(args.image_path, -1)
    img_height, img_width, img_channel = image.shape

    image_ref = cv2.imread(args.ref_path, -1)
    print(image.shape)
    print(image_ref.shape)



    genderClassfier = GenderToolClass(gender_prototxt_file_path, gender_net_file_path)
    img_ref_gender = genderClassfier.getGender(image_ref)
    print(img_ref_gender)



    dlib_landmark_predictor = dlib.shape_predictor(dlib_shape_model_path)
    swapface = FaceSwap(dlib_landmark_predictor)
    merge = swapface.swap_face(image_ref, image)
    cv2.imshow("merge", merge)
    cv2.waitKey(0)