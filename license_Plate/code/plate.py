from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2

# define the named tupled to store the license plate
LicensePlate = namedtuple("LicensePlateRegion", ["success", "plate", "thresh", "candidates"])

# define class
class LicensePlateDetector:
	def __init__(self, image, minPlateW=60, minPlateH=20, numChars=7, minCharW=40):
		# store the image to detect license plates in, the minimum width and height of the
		# license plate region, the number of characters to be detected in the license plate,
		# and the minimum width of the extracted characters
		self.image = image
		self.minPlateW = minPlateW
		self.minPlateH = minPlateH
		self.numChars = numChars
		self.minCharW = minCharW

	def detect(self):
		# detect license plate regions in the image
		lpRegions = self.detectPlates()

		# loop over the license plate regions
		for lpRegion in lpRegions:
			# detect character candidates in the current license plate region
			lp = self.detectCharacterCandidates(lpRegion)

			# only continue if characters were successfully detected
			if lp.success:
				# yield a tuple of the license plate object and bounding box
				yield (lp, lpRegion)

	def detectCharacterCandidates(self, region):
		# apply a 4-point transform to extract the license plate
		plate = perspective.four_point_transform(self.image, region)
		cv2.imshow("Perspective Transform", imutils.resize(plate, width=400))
		# extract the Value component from the HSV color space and apply adaptive thresholding
		# to reveal the characters on the license plate
		V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
		T = threshold_local(V, 29, offset=15, method="gaussian")
		thresh = (V > T).astype("uint8") * 255
		thresh = cv2.bitwise_not(thresh)

		# resize the license plate region to a canonical size
		plate = imutils.resize(plate, width=400)
		thresh = imutils.resize(thresh, width=400)
		cv2.imshow("Thresh", thresh)
		# perform a connected components analysis and initialize the mask to store the locations
		# of the character candidates
		labels = measure.label(thresh, neighbors=8, background=0)
		charCandidates = np.zeros(thresh.shape, dtype="uint8")
		# loop over the unique components
		for label in np.unique(labels):
			# if this is the background label, ignore it
			if label == 0:
				continue

			# otherwise, construct the label mask to display only connected components for the
			# current label, then find contours in the label mask
			labelMask = np.zeros(thresh.shape, dtype="uint8")
			labelMask[labels == label] = 255
			cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if imutils.is_cv2() else cnts[1]

			# ensure at least one contour was found in the mask
			if len(cnts) > 0:
				# grab the largest contour which corresponds to the component in the mask, then
				# grab the bounding box for the contour
				c = max(cnts, key=cv2.contourArea)
				(boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

				# compute the aspect ratio, solidity, and height ratio for the component
				aspectRatio = boxW / float(boxH)
				solidity = cv2.contourArea(c) / float(boxW * boxH)
				heightRatio = boxH / float(plate.shape[0])

				# determine if the aspect ratio, solidity, and height of the contour pass
				# the rules tests
				keepAspectRatio = aspectRatio < 1.0
				keepSolidity = solidity > 0.15
				keepHeight = heightRatio > 0.4 and heightRatio < 0.95

				# check to see if the component passes all the tests
				if keepAspectRatio and keepSolidity and keepHeight:
					# compute the convex hull of the contour and draw it on the character
					# candidates mask
					hull = cv2.convexHull(c)
					cv2.drawContours(charCandidates, [hull], -1, 255, -1)
				# clear pixels that touch the borders of the character candidates mask and detect
				# contours in the candidates mask
				charCandidates = segmentation.clear_border(charCandidates)

				# TODO:
				# There will be times when we detect more than the desired number of characters --
				# it would be wise to apply a method to 'prune' the unwanted characters

				# return the license plate region object containing the license plate, the thresholded
				# license plate, and the character candidates
				return LicensePlate(success=True, plate=plate, thresh=thresh,
									candidates=charCandidates)
