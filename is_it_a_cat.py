import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-c", "--cascade",
                default="haarcascade_frontalcatface.xml",
                help="path to cat detector haar cascade")
args = vars(ap.parse_args())
image = cv2.imread(args["image"]) # load image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert it to grayscale
# load the cat detector Haar cascade, then detect cat faces in image
detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(75, 75))
# minNeighbours- parameter controls the minimum number of detected bounding boxes
# in a given area for the region to be considered a “cat face”.
