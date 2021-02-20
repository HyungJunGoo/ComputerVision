from face_detect.face_detector import FaceDetector
import numpy as np
import cv2
import sys
import time
import threading
from os import listdir 
from os.path import isfile, join
import os.path
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from gui_app import QtImageViewer

face_detect_area_s_point = (480, 270)
face_detect_area_e_point = (800, 450)
area = (face_detect_area_s_point, face_detect_area_e_point)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
            
class Track_app():
    def __init__(self, section_count):
        super().__init__()
        self.face_x = -1
        self.term_of_section = int(1280/section_count)
        self.face_section = [x*self.term_of_section for x in range(section_count+1)]
        self.face_section.append(1280)
        self.current_face_section = 0

    def detect_face(self, frame):
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = FaceDetector()
        faceRects, num_faces = face_detector.detect(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        if num_faces == 0:
            return (), num_faces
        else:
            (x, y, w, h) = faceRects[0]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roiPts = (x, y, w, h)
            x_c = x+w/2
            y_c = y+h/2
            if( (x_c >= face_detect_area_s_point[0] and x_c <= face_detect_area_e_point[0]) and (y_c>=face_detect_area_s_point[1] and y_c<=face_detect_area_e_point[1])):
                return roiPts, num_faces
            else:
                return (), 0

    # Check if the tracking window is in the area
    def pts_in_area(self, pts, area):
        x_c = (pts[0][0] + pts[2][0])/2
        y_c = (pts[0][1] + pts[1][1])/2
        pts_h = abs(pts[0][1] - pts[1][1])
        pts_w = abs(pts[1][0] - pts[2][0])
        area_w = (area[0][0], area[1][0])
        area_h = (area[0][1], area[1][1])
        if( (x_c >= 50 and x_c <= 1230) and (y_c >= 50 and y_c<= 670)):
            return True
        return False

    def get_face_section(self):
        return self.current_face_section
    def set_face_section(self, i):
        self.current_face_section = i
        return

    def face_track(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        roi_hist = None
        t = 0
        while (True):
            ret, frame = cap.read()
            if ret == False:
                break
            if roi_hist is None:
                if t == 3:
                    roiPts, num_face = self.detect_face(frame)
                    if num_face != 0:
                        roi = frame[roiPts[1]:roiPts[1]+roiPts[3], roiPts[0]:roiPts[0]+roiPts[2]]
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
                        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                    t = 0 
                else:
                    t += 1
            else: # Face Track On
                # CamShift
                self.face_x = roiPts[0] + roiPts[2]/2
                for i, sect in enumerate(self.face_section):
                    if self.face_x >= sect and self.face_x <= self.face_section[i+1]:
                        # self.current_face_section = i
                        self.set_face_section(i)
                        break
                # print(f"x:{self.face_x}")
                print(self.current_face_section)

                roi_w = roiPts[2]
                roi_h = roiPts[3]
                scaled_roi_w = int(roi_w*1.5)
                scaled_roi_h = int(roi_h*1.5)
                n_x = int(roiPts[0] - 0.25*roi_w)
                n_y = int(roiPts[1] - 0.25*roi_h)
                if n_x < 0:
                    n_x = 0
                if n_y < 0:
                    n_y = 0;
                if n_x+scaled_roi_w> 1280:
                    n_x_2 = 1280
                else:
                    n_x_2 = n_x+scaled_roi_w
                
                if n_y + scaled_roi_h > 720:
                    n_y_2 = 720
                else:
                    n_y_2 = n_y + scaled_roi_h

                scaled_roiPts = [n_x, n_y, n_x_2, n_y_2]
                
                # Change to HSV Colour
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Histogram BackProjection
                dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
                roiPts = [n_x, n_y, scaled_roi_w, scaled_roi_h]
                ret, roiPts = cv2.CamShift(dst, roiPts, term_crit)
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                
                if self.pts_in_area(pts, area) is False:
                    roi_hist = None
                    t=0
                    face_x = -1
                    self.set_face_section(0)
                # else:
                #     cv2.polylines(frame, [pts], True, (0,255,255), 2)
            # cv2.imshow('frame', frame)
            k = cv2.waitKey(60) & 0xff
            if k == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()
        return
    
    def face_track_thread(self):
        thread = threading.Thread(target=self.face_track)
        thread.start()