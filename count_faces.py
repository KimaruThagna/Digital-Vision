import numpy as np
import cv2
# load cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('inputs/faces.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to greyscale