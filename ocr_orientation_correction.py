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
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)

    # Getting the median angle
    median_angle = np.median(angles)
    # Rotating the image with this median angle
    img_rotated = ndimage.rotate(img, median_angle)

    if save_image:
        cv2.imwrite('outputs/orientation_corrected.jpg', img_rotated)
    return img_rotated

img_rotated = orientation_correction(img_1)

# initializing the list for storing the coordinates
coordinates = []


# Defining the event listener (callback function)
def shape_selection(event, x, y, flags, param):
    # making coordinates global
    global coordinates
    # Storing the (x1,y1) coordinates when left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = [(x, y)] # grab starting point (top left of rectangle)
    elif event == cv2.EVENT_LBUTTONUP:
        coordinates.append((x, y)) # grab finishing point, bottom right of rectangle

        # Drawing a rectangle around the region of interest (roi) after mouse button is released
        cv2.rectangle(img_1, coordinates[0], coordinates[1], (0, 255, 255), 2)
        cv2.imshow("Region of interest", img_1)


# load the image and setup the mouse callback function
image_copy = img_rotated.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("image", image_copy)
    key = cv2.waitKey(1) & 0xFF

    if key==13: # If 'enter' is pressed, apply OCR
        break

    if key == ord("c"): # Clear the selection when 'c' is pressed
        image = image_copy.copy()

if len(coordinates) == 2: # top left and bottom right coords have been retrieved
    image_roi = image_copy[coordinates[0][1]:coordinates[1][1],
                coordinates[0][0]:coordinates[1][0]]
    cv2.imshow("Selected Region of Interest - Press any key to proceed", image_roi)
    extracted_text = pytesseract.image_to_string(image_roi)# alternatives include google or amazon vision
    print(f'The text in the selected region is :\n {extracted_text}')

    cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()
#perform OCR on the ROI
