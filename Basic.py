# import library
import cv2
import numpy as np
import face_recognition

# load img for training
imgKang = face_recognition.load_image_file('imageBasic/Kang Daniel.jpg')
imgKang = cv2.cvtColor(imgKang,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('imageBasic/ong seongwu.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# detect face return bounding box
faceLocation = face_recognition.face_locations(imgKang)[0]
faceLocationTest = face_recognition.face_locations(imgTest)[0]

# embedding face
encodeKang = face_recognition.face_encodings(imgKang)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

# show bounding box location left bottom, right top
cv2.rectangle(imgKang, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255,0,255),2)
cv2.rectangle(imgTest, (faceLocationTest[3], faceLocationTest[0]), (faceLocationTest[1], faceLocationTest[2]), (255,0,255),2)
# print(faceLocation)

# comparing image similarity
results = face_recognition.compare_faces([encodeKang], encodeTest)
faceDist = face_recognition.face_distance([encodeKang], encodeTest)
print(results, faceDist)

# put the result to the test image
cv2.putText(imgTest,f'{results} {round(faceDist[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,225),2)


# show image
cv2.imshow('Kang Daniel', imgKang)
cv2.imshow('Kang Test', imgTest)
cv2.waitKey(0)
