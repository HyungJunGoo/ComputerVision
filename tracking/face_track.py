from face_detect.face_detector import FaceDetector
import numpy as np
import cv2


casc_path = "cascade/haarcascade_frontalface_default.xml"


def FindFace(frame):

    allRoiPts = []
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = FaceDetector(casc_path)
    faceRects, num_faces = face_detector.detect(grey, scaleFactor=1.1, minNeighbors=5, minSize=(10,10))
    face_count = num_faces
    for (x, y, w, h) in faceRects:
        allRoiPts.append((x, y, x+w, y+h))

    return allRoiPts, face_count

def main():
    cap = cv2.VideoCapture(0)
    face_count = 0
    while face_count == 0:
        # take first frame of the video
        ret,frame = cap.read()

        # setup initial location of window
        track_windows, face_count = FindFace(frame)
        # set up the ROI for tracking

        all_roi = []
        for track_window in track_windows:
            roi = frame[track_window[1]:track_window[3], track_window[0]:track_window[2]]
            all_roi.append(roi)

    all_hsv_roi = []
    for roi in all_roi:
        all_hsv_roi.append(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    # hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    all_mask = []
    all_roi_hist = []
    for hsv_roi in all_hsv_roi:
        # mask = cv2.inRange(hsv_roi, np.array(0., 60., 32.)), np.array((180. , 255., 255.))
        # all_mask.append(mask)
        roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        all_roi_hist.append(roi_hist)
     # roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])

    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    while (1):
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            all_dst = []
            for roi_hist in all_roi_hist:
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                all_dst.append(dst)

            # dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            for i in range(0, face_count):
                ret, track_windows[i] = cv2.CamShift(all_dst[i], track_windows[i], term_crit)
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                cv2.polylines(frame, [pts], True, 255, 2)
                cv2.imshow("frame", frame)


            k = cv2.waitKey(60) & 0xff
            if k == ord('q'):
                break


        else:
            break


    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()