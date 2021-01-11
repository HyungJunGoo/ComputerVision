import cv2
import os

dataset_path = "./"
dataset_dir = os.listdir(dataset_path)
face_cascPath = "./haarcascade_frontalface_default.xml"
eyes_cascPath = "./haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyesCascade = cv2.CascadeClassifier(eyes_cascPath)


def face_dectection():
    print("face_detection started")
    video_capture = cv2.VideoCapture(0)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    # out = cv2.VideoWriter("result.mp4", fourcc, 20.0, size)
    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret is False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eyesCascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        # out.write(frame)
        # Display the resulting frame
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # When everything is done, release the capture
    video_capture.release()
    # out.release()
    cv2.destroyAllWindows()


face_dectection()
