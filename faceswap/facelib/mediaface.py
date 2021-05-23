"""
mediapip face
"""
import math
from typing import List, Tuple, Union

import argparse
import cv2
import mediapipe as mp
import numpy as np
import dlib

import faceswap.facelib.faceBlendCommon as fbc
from faceswap.faceswap_utils import ImageException

mp_face_detection = mp.solutions.face_detection

FACE_CONNECTIONS = ([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)
])

def getFaceCountour():
    return [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

def getFaces(image, min_detection_confidence=0.6):
    face_detection = mp.solutions.face_detection
    image_rows, image_cols, img_channel = image.shape
    faces = []
    bboxs = []
    with face_detection.FaceDetection(min_detection_confidence=min_detection_confidence) as f_detection:
        results = f_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for detection in results.detections:
            bbox_start_point = _normalized_to_pixel_coordinates(detection.location_data.relative_bounding_box.xmin,
                                                                detection.location_data.relative_bounding_box.ymin,
                                                                image_cols, image_rows)
            bbox_end_point = _normalized_to_pixel_coordinates(detection.location_data.relative_bounding_box.xmin+detection.location_data.relative_bounding_box.width,
                                                                detection.location_data.relative_bounding_box.ymin+detection.location_data.relative_bounding_box.height,
                                                                image_cols, image_rows)

            bboxs.append((bbox_start_point[0], bbox_start_point[1], bbox_end_point[0], bbox_end_point[1]))
            faces.append(image[bbox_start_point[1]:bbox_end_point[1], bbox_start_point[0]:bbox_end_point[0]])
    
    return faces, bboxs

def _normalized_to_pixel_coordinates(normalized_x: float,
                                     normalized_y:float,
                                     image_width: int,
                                     image_height: int) -> Union[None, Tuple[int, int]]:
    """将归一化的值转化为坐标值"""
    # 检查float值是否0~1.0
    def is_valid_normalized_value(value: float) -> bool:
        return(value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
        # TODO 此时超出了图像边界
        return None
    
    x_px = min(math.floor(normalized_x * image_width), image_width -1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def getFaceLandmarks(image, max_num_faces=1, min_detection_confidence=0.5, bounding_box=None, shape_predictor=None, face_bbox=None, isDlib=False):

    img_rows, img_cols, _ = image.shape
    faces_landmarks = []

    

    if isDlib:
        image_gray = image.copy()
        if image.shape[2] != 1 :
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmarks = shape_predictor(image_gray, face_bbox)
        points = []
        for p in landmarks.parts():
            pt = (p.x, p.y)
            points.append(pt)
        faces_landmarks.append(points)
        return faces_landmarks

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        else:
            for landmarks in results.multi_face_landmarks:
                one_face_landmarks = []
                for idx, landmark in enumerate(landmarks.landmark):
                    landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, img_cols, img_rows)
                    one_face_landmarks.append(landmark_px)
                faces_landmarks.append(one_face_landmarks)
        face_mesh = None
    return faces_landmarks


def faceSwap(image_template, image_ref, image_tempalted_landmarks, image_ref_landmarks,face_contour=None):
    image_tempalted_Warped = np.copy(image_template)

    # 计算hull
    hull_tempalted = []
    hull_ref = []
    
    # 如果没有指定合成区域的轮廓的话，则自动计算合成区域
    if not face_contour:
        hullIndex = cv2.convexHull(np.array(image_ref_landmarks), returnPoints=False)
        for idx in range(0, len(hullIndex)):
            hull_tempalted.append(image_tempalted_landmarks[hullIndex[idx][0]])
            hull_ref.append(image_ref_landmarks[hullIndex[idx][0]])
    else: # 使用指点的区域作为合成轮廓
        for idx in face_contour:
            hull_tempalted.append(image_tempalted_landmarks[face_contour[idx]])
            hull_ref.append(image_ref_landmarks[face_contour[idx]])
    
    
    

    sizeImage_ref = image_ref.shape
    rect = (0, 0, sizeImage_ref[1], sizeImage_ref[0])

    # delaunary三角形索引
    dt = fbc.calculateDelaunayTriangles(rect, hull_ref)

    if len(dt) == 0:
        raise ImageException(message="图片进行delaunary细分时出现错误!")
        return None
    
    # 对Delaunay三角形进行仿射变换
    # image_tempalted_Warped = image_tempalted_Warped.astype(np.float32)/255.0
    # image_ref = image_ref.astype(np.float32)/255.0
    for idx in range(0, len(dt)):
        t1 = []
        t2 = []

        #获取模板图像和参考图像上对应的三角形
        for j in range(0, 3):
            t1.append(hull_tempalted[dt[idx][j]])
            t2.append(hull_ref[dt[idx][j]])
        
        fbc.warpTriangle(image_tempalted_Warped, image_ref, t1, t2)
        # cv2.imshow("temp", image_tempalted_Warped)
        # cv2.waitKey(300)
        
    # 调整一下颜色
    # image_tempalted_Warped=fbc.correctColors(image_template, image_tempalted_Warped, np.array(image_tempalted_landmarks)[133], np.array(image_tempalted_landmarks)[336])
    # cv2.imshow("image_tempalted_Warped", image_tempalted_Warped)
    # cv2.waitKey(0)
    # 创建一个Mask for Seamless cloning
    hull8u = []
    for i in range(0, len(hull_tempalted)):
        hull8u.append((hull_tempalted[i][0], hull_tempalted[i][1]))
    mask = np.zeros(image_tempalted_Warped.shape, dtype=image_tempalted_Warped.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8u), (255, 255, 255))
    mask = cv2.GaussianBlur(mask,(3,3),10)
    # 对mask进行膨胀操作，将白色区域变大一点
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)


    # 获取目标图像中hull的中心点
    # mask = 255 * np.ones(image_template.shape, image_template.dtype)
    r = cv2.boundingRect(np.float32([hull_tempalted]))
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

    output = cv2.seamlessClone(np.uint8(image_tempalted_Warped), image_template, mask, center, cv2.NORMAL_CLONE)
    
    # cv2.imshow("output", output)

    # cv2.waitKey(0)
    return output
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meadiapip face swap')
    parser.add_argument('-i', '--image_path', default='./res/template1/template_F.png', type=str,
                        help='path to input image')
    parser.add_argument('-r', '--ref_path', default='./res/img1.jpeg', type=str, 
                        help='path to reference image(texture ref)')
    parser.add_argument('-m', '--mask_path', default=None, type=str, 
                        help='path to reference image(texture ref)')
    args = parser.parse_args()
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    img_height, img_width, img_channel = image.shape

    image_ref = cv2.imread(args.ref_path, cv2.IMREAD_COLOR)

    print(image.shape)
    print(image_ref.shape)

    if args.mask_path:
        image_hair = cv2.imread(args.mask_path, cv2.IMREAD_COLOR)
        image_hair = cv2.resize(image_hair, (image.shape[1], image.shape[0]))
        print(image_hair.shape)

        image_hair_gray = cv2.cvtColor(image_hair, cv2.COLOR_BGR2GRAY)
        _, image_hair_mask = cv2.threshold(image_hair_gray, 127, 255, cv2.THRESH_BINARY_INV)

        image_bk = cv2.bitwise_or(image, image, mask=(image_hair_mask-255))
        # cv2.imshow("bk", image_bk)
        image_fk = cv2.bitwise_or(image_hair, image_hair, mask=image_hair_mask)
        # cv2.imshow("fk", image_fk)
        image_with_hair = cv2.bitwise_or(image_bk, image_fk)
        # cv2.imwrite("./res/template21_with_hair.jpg", )
        # cv2.imshow("image_with_hair", image_with_hair)
        # cv2.waitKey(0)

    # # print(img_height, img_width, img_channel)

    # # image_ref_landmarks = getFaceLandmarks(image_ref, max_num_faces=1)[0]
    # image_tempalted_landmarks = getFaceLandmarks(image, max_num_faces=1)[0]

    # image_template_anno = image.copy()
    # for point in image_tempalted_landmarks:
    #     cv2.circle(image_template_anno, point, 2, (0, 255, 0), 2)
    # cv2.imshow("image_template_anno", image_template_anno)

    # # image_ref_anno = image_ref.copy()
    # # for point in image_ref_landmarks:
    # #     cv2.circle(image_ref_anno, point, 2, (0, 255, 0), 2)
    # # cv2.imshow("image_ref_anno", image_ref_anno)
    # # cv2.waitKey(0)

    # mp_face_mesh = mp.solutions.face_mesh
    # mp_drawing = mp.solutions.drawing_utils
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # annotated_image = image_ref.copy()
    # with mp_face_mesh.FaceMesh(
    #                         static_image_mode=True,
    #                         max_num_faces=1,
    #                         min_detection_confidence=0.5) as face_mesh:
    #     # Convert the BGR image to RGB before processing.
    #     results = face_mesh.process(cv2.cvtColor(image_ref, cv2.COLOR_BGR2RGB))

    #     # Print and draw face mesh landmarks on the image.
    #     if not results.multi_face_landmarks:
    #          pass
        
    #     for face_landmarks in results.multi_face_landmarks:
    #         mp_drawing.draw_landmarks(
    #                             image=annotated_image,
    #                             landmark_list=face_landmarks,
    #                             connections=mp_face_mesh.FACE_CONNECTIONS,
    #                             landmark_drawing_spec=drawing_spec,
    #                             connection_drawing_spec=drawing_spec)
    
    # cv2.imshow("aaa", annotated_image)
    # cv2.waitKey(0)


    # facelandmarks
    # mp_face_mesh = mp.solutions.face_mesh
    # annotated_image = image.copy()
    # with mp_face_mesh.FaceMesh(
    #     static_image_mode=True,
    #     max_num_faces=1,
    #     min_detection_confidence=0.5) as face_mesh:
    #     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
    #     if not results.multi_face_landmarks:
    #         pass
    #     else:
    #         for face_landmarks in results.multi_face_landmarks:
    #             # print('face_landmarks:', face_landmarks)
    #             for idx, landmark in enumerate(face_landmarks.landmark):
    #                 landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, img_width, img_height)
    #                 cv2.putText(annotated_image, str(idx), landmark_px, cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0))
    #                 cv2.circle(annotated_image, landmark_px, 1, (0, 255, 0), 1)

    # annotated_image = image.copy()
    # faces_landmarks = getFaceLandmarks(image)
    # for point in getFaceCountour():
    #     cv2.circle(annotated_image, faces_landmarks[0][point], 2, (0, 255, 0), 2)


    # cv2.imshow("landmarks", annotated_image)

    # annotated_image1 = image_ref.copy()
    # faces_landmarks = getFaceLandmarks(image_ref)
    # for point in getFaceCountour():
    #     cv2.circle(annotated_image1, faces_landmarks[0][point], 2, (0, 255, 0), 2)


    # cv2.imshow("landmarks_ref", annotated_image)
    # cv2.waitKey(0)
    # cv2.imwrite("./landmarks.jpg", annotated_image)

    # faceSwap(image, image_ref)
    # image_ref_landmarks = getFaceLandmarks(image_ref, max_num_faces=5)
    # image_ref_anno = image_ref.copy()
    # for landmarks in image_ref_landmarks:
    #     for point in landmarks:
    #         cv2.circle(image_ref_anno, point, 2, (0, 255, 0), 2)
    # cv2.imshow("image_ref_anno", image_ref_anno)

    # cv2.waitKey(0)

    landmark_detector = dlib.shape_predictor("./res/dlib/shape_predictor_68_face_landmarks.dat")
    # faces, faces_bboxs = getFaces(image)
    if args.mask_path:
        image_prossed = image_with_hair.copy()
    else:
        image_prossed = image.copy()

    faces, faces_bboxs = getFaces(image_prossed)
    faces_ref, faces_ref_bboxs = getFaces(image_ref)
    
    # bbox = dlib.rectangle(faces_bboxs[0][0], faces_bboxs[0][1], faces_bboxs[0][2], faces_bboxs[0][3])
    # bbox = dlib.rectangle(*faces_bboxs[0])
    # image_tempalte_landmarks = getFaceLandmarks(image_with_hair, shape_predictor=landmark_detector, face_bbox=bbox, isDlib=True)[0]
    image_tempalte_landmarks = getFaceLandmarks(image_prossed)[0]
    # image_tempalte_landmarks = getFaceLandmarks(image_prossed)[0]

    # bbox = [faces_ref_bboxs[0][0], faces_ref_bboxs[0][1], faces_ref_bboxs[0][2], faces_ref_bboxs[0][3]]
    # bbox = dlib.rectangle(*faces_ref_bboxs[0])
    # image_ref_landmarks = getFaceLandmarks(image_ref, shape_predictor=landmark_detector, face_bbox=bbox, isDlib=True)[0]
    image_ref_landmarks = getFaceLandmarks(image_ref)[0]


    annotated_image = image_prossed.copy()
    for point in image_tempalte_landmarks:
        cv2.circle(annotated_image, point, 2, (0, 255, 0), 2)
    bbox = faces_bboxs[0]
    cv2.rectangle(annotated_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=2)
    cv2.imshow("landmarks", annotated_image)

    annotated_image1 = image_ref.copy()
    for point in image_ref_landmarks:
        cv2.circle(annotated_image1, point, 2, (0, 255, 0), 2)
    bbox = faces_ref_bboxs[0]
    cv2.rectangle(annotated_image1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), thickness=2)
    cv2.imshow("landmarks_ref", annotated_image1)
    # cv2.waitKey(0)


    output = faceSwap(image, image_ref, image_tempalte_landmarks, image_ref_landmarks)

    if args.mask_path:
        output_bk = cv2.bitwise_or(output, output, mask=(image_hair_mask-255))
        # cv2.imshow("bk", image_bk)
        output_fk = cv2.bitwise_or(image_hair, image_hair, mask=image_hair_mask)
        # cv2.imshow("fk", image_fk)
        output = cv2.bitwise_or(output_bk, output_fk)

    cv2.imshow("faceSwap", output)
    cv2.waitKey(0)
