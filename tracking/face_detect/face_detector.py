import cv2

class FaceDetector:
    def __init__(self):
        # load the face detector cascade
        self.faceCascade = cv2.CascadeClassifier("/Users/hyungjungu/Documents/Project/CV/project/tracking/face_detect/haarcascade_frontalface_default.xml")
        # self.faceCascade = cv2.CascadeClassifier("/Users/hyungjungu/Documents/Project/CV/project/tracking/face_detect/haarcascade_frontalface_alt2.xml")

    def detect(self, image, scaleFactor=1.2, minNeighbors=5, minSize=(40,40)):
        # detect faces in the image
        rects = self.faceCascade.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        number_of_face = len(rects)
        # return the rectangles representing bounding
        # boxes around the faces
        return rects, number_of_face
