from typing import Tuple
import mediapipe as mp
import dlib
import cv2
import numpy as np
import argparse
import math

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

                landmarks = self.__dlib_shape_to_array(shape)
                success, rotationVector, translationVector = self.getHeadPose(image, landmarks)
                # vec = [_[0] for _ in rotationVector]
                # vec = [math.degrees(math.asin(math.sin(_))) for _ in vec]
                # print(vec)
                if success:
                    # rotation_M = cv2.Rodrigues(rotationVector)[0]
                    # P = np.hstack((rotation_M, np.zeros((3, 1), dtype=np.float64)))
                    # eulerAngles = cv2.decomposeProjectionMatrix(P)[6]
                    rotation_mat, _ = cv2.Rodrigues(rotationVector) #旋转矩阵
                    eulerAngles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_mat)
                    # print(eulerAngles)
                    # pose_mat = cv2.hconcat((rotation_mat, translationVector)) #投影矩阵
                    # _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)

                    rotation = dict()

                    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
                    
                    rotation['pitch'] = math.degrees(math.asin(math.sin(pitch)))
                    rotation['yaw'] = math.degrees(math.asin(math.sin(yaw)))
                    rotation['roll'] = -math.degrees(math.asin(math.sin(roll)))
                    

                    # angle_x = rotationVector[0][0]*180/np.pi
                    # angle_y = rotationVector[1][0]*180/np.pi
                    # angle_z = rotationVector[2][0]*180/np.pi
                    angles.append(rotation)
                else:
                    angles.append(None)

                # # 计算水平旋转方向
                # left_eye_inner = np.array([shape.part(39).x, shape.part(39).y])
                # right_eye_inner = np.array([shape.part(42).x, shape.part(42).y])
                # eye_vec = right_eye_inner - left_eye_inner
                # ref_vec = np.array([image.shape[1], 0])
                # cosangle = eye_vec.dot(ref_vec)/(np.linalg.norm(eye_vec) * np.linalg.norm(ref_vec))
                # angle_h = np.abs((np.arccos(cosangle)*180.0)/np.pi)
                # # 计算数值旋转方向
                # nose_bottom = np.array([shape.part(27).x, shape.part(27).y])
                # nose_top = np.array([shape.part(30).x, shape.part(30).y])
                # nose_vec = nose_top - nose_bottom
                # ref_vec = np.array([0, image.shape[0]])
                # cosangle = nose_vec.dot(ref_vec)/(np.linalg.norm(nose_vec) * np.linalg.norm(ref_vec))
                # angle_v = np.abs((np.arccos(cosangle)*180.0)/np.pi)
                # angles.append((angle_h, angle_v))
        return bbox, angles

    def getHeadPose(self, image, landmarks):
        """计算头部的3D姿态"""
        imagePoints = np.array([landmarks[30],  # noses tip
                                landmarks[8],   # chin
                                landmarks[36],  # left eye corner
                                landmarks[45],  # right eye corner
                                landmarks[48],  # left mouth corner
                                landmarks[54]], # right mouth corner
                                dtype=np.float64)
        modelPoints = np.array([[0.0, 0.0, 0.0],
                                [0.0, -330.0, -65.0],
                                [-225.0, 170.0, -135.0],
                                [225.0, 170.0, -135.0],
                                [-150.0, -150.0, -125.0],
                                [150.0, -150.0, -125.0]], 
                                dtype=np.float64)
        
        # 将摄像头当做无畸变摄像头处理
        height, width, ch = image.shape
        focalLength = width
        center = (height/2, width/2)
        cameraMatrix = self.getCameraMatrix(focalLength, center)
        # 这里假设假设摄像头是无畸变的
        distCoeffs = np.zeros((4, 1), dtype=np.float64)
        #使用solvePnP解算器计算头部的转动向量和移动向量
        success, rotationVector, translationVector = cv2.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs)

        # # 绘制一下坐标轴
        # noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
        # noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, distCoeffs)

        # p1 = (int(imagePoints[0, 0]), int(imagePoints[0, 1]))
        # p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))

        # cv2.arrowedLine(image, p1, p2, (255, 0, 0), thickness=2)

        # # self.drawHeadPoseAxies(image, rotationVector, translationVector)


        # cv2.imshow("image pose", image)
        # cv2.waitKey()

        return success, rotationVector, translationVector
    
    def drawHeadPoseAxies(self, image: np.ndarray, 
                          rotation: np.ndarray, 
                          translation: np.ndarray, 
                          focal_length: Tuple[float, float]=(1.0, 1.0), 
                          principal_point: Tuple[float, float]=(0.0, 0.0),
                          axis_length: float = 0.1):
        image_rows, image_cols, _ = image.shape
        axis_world = np.float32([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        axis_cam = np.matmul(rotation, axis_length*axis_world.T).T + translation
        x = axis_cam[..., 0]
        y = axis_cam[..., 1]
        z = axis_cam[..., 2]
        fx, fy = focal_length
        px, py = principal_point
        x_ndc = np.clip(-fx * x / (z + 1e-5) + px, -1., 1.)
        y_ndc = np.clip(-fy * y / (z + 1e-5) + py, -1., 1.)
        x_im = np.int32((1 + x_ndc) * 0.5 * image_cols)
        y_im = np.int32((1 - y_ndc) * 0.5 * image_rows)
        origin = (x_im[0], y_im[0])
        x_axis = (x_im[1], y_im[1])
        y_axis = (x_im[2], y_im[2])
        z_axis = (x_im[3], y_im[3])
        cv2.arrowedLine(image, origin, x_axis, (0, 0, 255),2)
        cv2.arrowedLine(image, origin, y_axis, (0, 255, 0),2)
        cv2.arrowedLine(image, origin, z_axis, (0, 255, 0),2)



    def getCameraMatrix(self, focalLength, center):
        """计算摄像头的变换矩阵"""
        cameraMatrix = np.array([[focalLength, 0, center[0]],
                                [0, focalLength, center[1]],
                                [0, 0, 1]], dtype=np.float64)
        return cameraMatrix

    
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
