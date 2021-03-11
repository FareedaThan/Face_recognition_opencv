# import library
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# import all images
path = 'imageAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for imgName in myList:
    currentImg = cv2.imread(f'{path}/{imgName}')
    images.append(currentImg)
    classNames.append(os.path.splitext(imgName)[0])

print(classNames)

# encode the image reference
def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# add name and time attendance arrived
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        # prevent repeating name list
        if name not in nameList:
            now = datetime.now()
            dateString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dateString}')



encodeListKnown = findEncoding(images)

# get the input image to compare
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    # scale down 4 times
    imgInput = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgInput = cv2.cvtColor(imgInput,cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgInput)
    encodesCurFrame = face_recognition.face_encodings(imgInput, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # matching with the lowest(distance) element value
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            # increase 4 time to actual value
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1),(x2,y2), (0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255),2)
            markAttendance(name)

    # show the image from webcam
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)














