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