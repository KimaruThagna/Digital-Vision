from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file") # if present, go for video else, use webcam
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size") # number of coords of previous object locations to be stored
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green" object in the HSV color space, then initialize the
# list of tracked points
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
tracked_points = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start() # start webcam
else: # otherwise, grab a reference to the video file
    vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)
# infinite loop

while True:
    frame = vs.read() # grab the current frame
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame
    # if we are viewing a video and we did not grab a frame, then we have reached the end of the video
    if frame is None:
        break

    frame = imutils.resize(frame, width=600) # resize the frame,
    blurred = cv2.GaussianBlur(frame, (11, 11), 0) # blur frame
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # convert to HSV

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2) # remove boundaries in region of interest hence make smaller
    mask = cv2.dilate(mask, None, iterations=2) # opposite of erosion

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    tracked_points.appendleft(center)