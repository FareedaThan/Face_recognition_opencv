# import library
import cv2
import numpy as np
import face_recognition

# load img for training
imgKang = face_recognition.load_image_file('imageBasic/Kang Daniel.jpg')
imgKang = cv2.cvtColor(imgKang,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('imageBasic/Kang test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# show image
cv2.imshow('Kang Daniel', imgKang)
cv2.imshow('Kang Test', imgTest)
cv2.waitKey(0)
