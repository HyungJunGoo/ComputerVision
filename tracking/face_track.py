from face_detect.face_detector import FaceDetector
import numpy as np
import cv2


casc_path = "cascade/haarcascade_frontalface_default.xml"
face_detect_area_s_point = (320, 120)
face_detect_area_e_point = (960, 600)
area = (face_detect_area_s_point, face_detect_area_e_point)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

def FindFace(frame):

    allRoiPts = []
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = FaceDetector(casc_path)
    faceRects, num_faces = face_detector.detect(grey, scaleFactor=1.1, minNeighbors=5, minSize=(10,10))
    face_count = num_faces
    if num_faces == 0:
        return (), face_count
    else:
        (x, y, w, h) = faceRects[0]
        roiPts = (x, y, x+w, y+h)
        return roiPts, face_count

def pts_in_area(pts, area):
    print(f"pts : {pts}")
    x_c = (pts[0][0] + pts[2][0])/2
    y_c = (pts[0][1] + pts[1][1])/2
    pts_h = abs(pts[0][1] - pts[1][1])
    pts_w = abs(pts[1][0] - pts[2][0])

    area_w = (area[0][0], area[1][0])
    area_h = (area[0][1], area[1][1])
    if( (x_c <= area_w[1] and x_c >= area_w[0]) and (y_c <= area_h[1] and y_c >= area_h[0])):
        return True
    
    return False

def on_track(cap, track_window, roi_hist):
    while (1):
        ret, frame = cap.read()
        frame = cv2.rectangle(frame, face_detect_area_s_point, face_detect_area_e_point, (0,150,255), 1)
        cv2.imshow("frame", frame)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
    
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        if pts_in_area(pts, area) is False:
            break
        cv2.polylines(frame, [pts], True, (0,255,255), 2)
        cv2.imshow("frame", frame)
        k = cv2.waitKey(60) & 0xff
        if k == ord('q'):
            break
    return False
        
def set_roi(cap):
    face_count = 0
    while face_count == 0:
        print("Finding Face")
        # take first frame of the video
        ret, frame = cap.read()
        frame = cv2.rectangle(frame, face_detect_area_s_point, face_detect_area_e_point, (0,150,255), 1)
        cv2.imshow("frame", frame)
        # setup initial location of window
        track_window, face_count = FindFace(frame)
    # set up the ROI for tracking
    # print(f"track window : {track_window}")    
    roi = frame[track_window[1]:track_window[3], track_window[0]:track_window[2]]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return track_window, roi_hist

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
    
    track_window, roi_hist = set_roi(cap)
    track_on = True

    while (1):
        ret, frame = cap.read()
        frame = cv2.rectangle(frame, face_detect_area_s_point, face_detect_area_e_point, (0,150,255), 1)
        if track_on:
            track_on = on_track(cap, track_window, roi_hist)
        else:
            set_roi(cap)
            track_on = True

        k = cv2.waitKey(60) & 0xff
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()