import cv2
import os

dataset_path = "./"
dataset_dir = os.listdir(dataset_path)
face_cascPath = "./haarcascade_frontalface_default.xml"
eyes_cascPath = "./haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyesCascade = cv2.CascadeClassifier(eyes_cascPath)


class FaceDetector:
    def __init__(self, faceCascadePath):
        # load the face detector cascade
        self.faceCascade = cv2.CascadeClassifier(faceCascadePath)

    def detect(self, image, scaleFactor=1.2, minNeighbors=5, minSize=(30,30)):
        # detect faces in the image
        rects = self.faceCascade.detectMultiScale(image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
        number_of_face = len(rects)
        # return the rectangles representing bounding
        # boxes around the faces
        return rects, number_of_face