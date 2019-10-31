import numpy as np
import cv2
# load cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('inputs/faces.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to greyscale
# actual face detection
faces = face_cascade.detectMultiScale(gray_img)
if len(faces) == 0:
    print( "No faces found")
else:
    print(faces)
    print(faces.shape)
    print(f'Number of faces detected: {faces.shape[0]}')

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1) # green boxes
        
cv2.rectangle(img, ((0,img.shape[0] -25)),(270, img.shape[0]), (255,255,255), -1) #create rectangle to display text
cv2.putText(img, "I have seen: " + str(faces.shape[0]), (0,img.shape[0] -10)+"faces", cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1)