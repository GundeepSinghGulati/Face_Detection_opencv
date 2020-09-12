# -*- coding: utf-8 -*-
"""
@author: Gundeep Gulati
"""
#face recognition using open cv library

#import the library
import cv2

#loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#functions which define detections on face or eyes
def detect(gray, frame):
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w] #roi = region of intrest in black and white image
        roi_color = frame[y:y+h, x:x+w] #frame holds the original color image
        eye = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
        #eyes detecting loop inside the face
        for (ex, ey, ew, eh) in eye:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2) #ex = eyes coordinates inside the face
    return frame
#doing face recognition using webcam
video_capture = cv2.VideoCapture(0) #0 for internal camera 1 for external camera
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame) 
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release() #turn off the camera
cv2.destroyAllWindows()
       