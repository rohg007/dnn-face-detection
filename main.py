import cv2
import numpy as np
import time
# import dlib

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
overlay = cv2.imread('nft.jpg')
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# hogFaceDetector = dlib.get_frontal_face_detector()
video = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    check, frame = video.read()
    ##Haar Cascade
    # faces = face_cascade.detectMultiScale(frame,
    #                                       scaleFactor=1.1, minNeighbors=5)
    # for x,y,w,h in faces:
    #     overlay = cv2.resize(overlay, (w,h))
    #     frame[y:y+h, x:x+w] = overlay

    h, w = frame.shape[:2]
    start = time.time()
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    end = time.time()
    start_drawing = time.time()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            # cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
            width, height = x1-x, y1-y
            overlay = cv2.resize(overlay, (width,height))
            frame[y:y1, x:x1] = overlay
    end_drawing = time.time()
    fps_label = "FPS: %.2f (excluding %.2fms draw time)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    # out.write(frame)
    cv2.imshow('Face Detector', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()