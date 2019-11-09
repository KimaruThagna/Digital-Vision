import numpy as np
import cv2
import math
from scipy import ndimage
import pytesseract
# read test images
img_1 = cv2.imread("inputs/ocr_image.jpg")
img_2 = cv2.imread("inputs/ocr_image_2.jpg")

'''
Adjust orientation by using the Canny(detect edges) and HoughLines(detect lines) algorithms and open cv
'''
def orientation_correction(img, save_image=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # image to greyscale
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3) # detect edges
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)# detect lines
    angles = []
    for x1, y1, x2, y2 in lines[0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Getting the median angle
    median_angle = np.median(angles)
    print(angles)
    # Rotating the image with this median angle
    img_rotated = ndimage.rotate(img, median_angle)

    if save_image:
        cv2.imwrite('orientation_corrected.jpg', img_rotated)
    return img_rotated

img_rotated = orientation_correction(img_1)