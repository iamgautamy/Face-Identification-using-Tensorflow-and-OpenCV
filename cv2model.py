import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
import time
import tensorflow as tf
import os

face_cascade = cv2.CascadeClassifier(r'D:\self_project_frameworks\face_detection_2022\haarcascade_frontalface_default.xml') 
detector = tf.keras.models.load_model('my_model.h5')
classes = ['face1', 'face10', 'face11', 'face12', 'face13', 'face14', 'face15', 'face16', 'face2', 'face3', 'face4', 'face5', 'face6', 'face7', 'face8', 'face9']
width = 180
height= 180
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    img = cv2.resize(frame,(width,height))
    rgb =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(rgb)

    #Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    #Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
    pred = detector.predict(rgb_tensor)
    pred = np.argmax(pred)
    label = classes[pred]

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = img[y:y+h, x:x+w]
        cv2.putText(img,label,org = (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 255), thickness = 2)

    #define the screen resulation
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    #resized window width and height
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    #cv2.WINDOW_NORMAL makes the output window resizealbe
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Window', window_width, window_height)

    cv2.imshow('Resized Window', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
  
# De-allocate any associated memory usage
cv2.destroyAllWindows() 