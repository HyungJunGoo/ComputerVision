from face_detect.face_detector import FaceDetector
import numpy as np
import cv2

face_detect_area_s_point = (480, 270)
face_detect_area_e_point = (800, 450)
area = (face_detect_area_s_point, face_detect_area_e_point)
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

def detect_face(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #face_detector = FaceDetector(casc_path)
    face_detector = FaceDetector()
    faceRects, num_faces = face_detector.detect(grey, scaleFactor=1.1, minNeighbors=5, minSize=(10,10))
    
    if num_faces == 0:
        return (), num_faces
    else:
        (x, y, w, h) = faceRects[0]
        roiPts = (x, y, x+w, y+h)
        x_c = x+w/2
        y_c = y+h/2
        print(f"face xc,yc : {x_c, y_c}")
        # if ( (x >= face_detect_area_s_point[0] and x+w <= face_detect_area_e_point[0]) and (y>=face_detect_area_s_point[1] and y+h <= face_detect_area_e_point[1]) ):
        #     return roiPts, num_faces
        if( (x_c >= face_detect_area_s_point[0] and x_c <= face_detect_area_e_point[0]) and (y_c>=face_detect_area_s_point[1] and y_c<=face_detect_area_e_point[1])):
            return roiPts, num_faces
        else:
            return (), 0

# Check if the tracking window is in the area
def pts_in_area(pts, area):
    print(f"pts : {pts}")
    x_c = (pts[0][0] + pts[2][0])/2
    y_c = (pts[0][1] + pts[1][1])/2
    print(f"xc, yc: {x_c, y_c}")
    pts_h = abs(pts[0][1] - pts[1][1])
    pts_w = abs(pts[1][0] - pts[2][0])

    area_w = (area[0][0], area[1][0])
    area_h = (area[0][1], area[1][1])
    if( (x_c >= 0 and x_c <= 1280) and (y_c >= 0 and y_c<= 720)):
        return True
    
    return False

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cv2.namedWindow("frame")
    roi_hist = None
    t = 0
    while (True):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.rectangle(frame, face_detect_area_s_point, face_detect_area_e_point, (0,150,255), 1)
        
        if roi_hist is None:
            if t == 3:
                print("finding face")
                roiPts, num_face = detect_face(frame)
                # print(f"roiPts : {roiPts}")
                if num_face != 0:
                    roi = frame[roiPts[1]:roiPts[3], roiPts[0]:roiPts[2]]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
                    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                t = 0 
            else:
                t += 1
            # print(f"roi: {roi}")
        else:
            
            # CamShift
            # Todo -> frame size
            # pts must be keep tracked
            print(f"roipts: {roiPts}")
            # searchPts = 
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
            
            
            ret, roiPts = cv2.CamShift(dst, roiPts, term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            if pts_in_area(pts, area) is False:
                roi_hist = None
                t=0
            else:
                cv2.polylines(frame, [pts], True, (0,255,255), 2)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(60) & 0xff
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    return

if __name__ == "__main__":
    main()
