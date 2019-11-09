import numpy as np
import cv2
import math
from scipy import ndimage
import pytesseract
# read test images
img_1 = cv2.imread("inputs/ocr_image.jpg")
img_2 = cv2.imread("inputs/ocr_image_2.jpg")