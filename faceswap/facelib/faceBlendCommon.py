"""
面部合成的通用方法
"""

import cv2
import numpy as np

def rectContains(rect, point):
    """
        检测点是否在矩形内
    """
    
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def calculateDelaunayTriangles(rect, points):
    """
        计算点积之间的Delanay三角形
    Return:
        返回每个三角形的顶点的索引
    """

    # 创建一个细分实例
    subdiv = cv2.Subdiv2D(rect)

    # 插入细分的点
    for p in points:
        subdiv.insert((p[0], p[1]))
    # 获取细分三角形
    triangleList = subdiv.getTriangleList()

    # 获取细分三角形的顶点索引
    delaunaryTri = []
    for t in triangleList:
        # getTriangleList方法返回的是三角形的
        # 三个顶点的坐标: x1, y1, x2, y2, x3, y3
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            # 储存三角形的点索引
            ind = []
            # 获取每个点在landmark中的索引
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            # 以索引的形式存储三角形
            if len(ind) == 3:
                delaunaryTri.append((ind[0], ind[1], ind[2]))
    return delaunaryTri

def applyAffineTransform(src, srcTri, dstTri, size):
    """
    使用srcTri到dstTri的计算的仿射变化,size为输出的图像的大小
    """
    

    # 计算仿射变换
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # 应用仿射变化
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst

def warpTriangle(img1, img2, t1, t2):
    """
        将img2的区域变化到img1上面，并返回变换后的image1
    """
    

    # 计算每个三角形的包括矩形
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    

    #使用left-top点作为新的坐标原点
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        # t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
    
    # 获取填充三角形的mask
    mask = np.zeros((r1[3], r1[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t1Rect), (1.0, 1.0, 1.0), 16, 0)

    # 在三角形区域使用仿射变换
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    size = (r1[2], r1[3])
    img1Rect = applyAffineTransform(img2Rect, t2Rect, t1Rect, size)

    img1Rect = img1Rect * mask

    # 拷贝变换区域到输出图像上
    img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] * ((1.0, 1.0, 1.0) - mask)
    img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] + img1Rect