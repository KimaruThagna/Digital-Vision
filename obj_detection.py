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
counter = 0
(dX, dY) = (0, 0)
direction = ""

# if a video path was not supplied, grab the reference to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start() # start webcam


else: # otherwise, grab a reference to the video file
    vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)
# infinite loop
vid_cod = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter("cam_video.mp4", vid_cod, 20.0, (640, 480))
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

    # find contours in the mask and initialize the current (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            cv2.circle(frame, (int(x), int(y)), int(radius), # draw the circle and centroid on the frame,
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1) # centroid

    # update the points queue
    tracked_points.appendleft(center)
    # loop over the set of tracked points
    for i in np.arange(1, len(tracked_points)):
        # if either of the tracked points are None, ignore

        if tracked_points[i - 1] is None or tracked_points[i] is None:
            continue

        # check to see if enough points have been accumulated in buffer
        if counter >= 10 and i == 1 and tracked_points[-10] is not None:
            # compute the difference between the x and y
            # coordinates and re-initialize the direction
            # text variables
            dX = tracked_points[-10][0] - tracked_points[i][0]
            dY = tracked_points[-10][1] - tracked_points[i][1]
            (dirX, dirY) = ("", "")

            # ensure there is significant movement in the
            # x-direction
            if np.abs(dX) > 20:
                dirX = "East" if np.sign(dX) == 1 else "West"

            # ensure there is significant movement in the
            # y-direction
            if np.abs(dY) > 20:
                dirY = "North" if np.sign(dY) == 1 else "South"

            # handle when both directions are non-empty
            if dirX != "" and dirY != "":
                direction = f'{dirY}-{dirX}'
            else:# otherwise, only one direction is non-empty
                direction = dirX if dirX != "" else dirY
    # loop over the set of tracked points
    for i in range(1, len(tracked_points)):
        # if either of the tracked points are None, ignore them
        if tracked_points[i - 1] is None or tracked_points[i] is None:
            continue

        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5) # most recent item is thicker
        cv2.line(frame, tracked_points[i - 1], tracked_points[i], (255, 0, 0), thickness)
        # show the movement deltas and the direction of movement on frame
        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,  0.65, (0, 255, 255), 3)
        cv2.putText(frame, f'dx: {dX}, dy: {dY}', (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,   0.35, (0, 255, 0), 1)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    counter+=1

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

else:# otherwise, release the camera
    vs.release()

# close all windows
output.release()

cv2.destroyAllWindows()