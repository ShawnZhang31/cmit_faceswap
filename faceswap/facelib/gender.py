import dlib
import cv2


class GenderToolClass():
    def __init__(self, proto_file_path, model_file_path):
        self.genderNet = cv2.dnn.readNetFromCaffe(proto_file_path, model_file_path)
        self.genderList = ['Male', 'Female']
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # 使用Dlib获取的面部区域
    def getFaceBox(self, image):
        frameDlib = image.copy()
        faceDetector = dlib.get_frontal_face_detector()
        bboxes = []
        faces = faceDetector(frameDlib, 0)
        for face in faces:
            bboxes.append([face.left(), face.top(), face.right(), face.bottom()])
        return bboxes
    
    def getGender(self, image):
        bboxes = self.getFaceBox(image)
        box = bboxes[0]
        face = image[max(0,box[1]):min(box[3],image.shape[0]-1),max(0,box[0]):min(box[2], image.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.genderNet.setInput(blob)
        genderPreds = self.genderNet.forward()
        return self.genderList[genderPreds[0].argmax()]

