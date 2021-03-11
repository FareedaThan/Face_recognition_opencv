# import library
import cv2
import numpy as np
import face_recognition

# load img for training
imgKang = face_recognition.load_image_file('imageBasic/Kang Daniel.jpg')
imgKang = cv2.cvtColor(imgKang,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('imageBasic/Kang test.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# detect face return bounding box
faceLocation = face_recognition.face_locations(imgKang)[0]


# embedding face
encodeKang = face_recognition.face_encodings(imgKang)[0]

# show bounding box location left bottom, right top
cv2.rectangle(imgKang, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255,0,255),2)
print(faceLocation)


# show image
cv2.imshow('Kang Daniel', imgKang)
cv2.imshow('Kang Test', imgTest)
cv2.waitKey(0)
