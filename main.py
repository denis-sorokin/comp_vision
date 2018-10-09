import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

imagePath = sys.argv[1]

# load the image and convert it to grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# compute the Scharr gradient magnitude representation of the images
# in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = 5)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = 5)
 
# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur and threshold the image
blurred = cv2.blur(gradient, (10, 10))
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.dilate(image, kernel)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
img, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)
 
# compute the rotated bounding box of the largest contour
for el in c:
	rect = (cv2.minAreaRect(el))
	box = (np.int0(cv2.boxPoints(rect)))
 
	# draw a bounding box arounded the detected barcode and display the
	# image
	cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

resized_image = cv2.resize(image, (1280, 720)) 
cv2.imshow('gradient', resized_image)

cv2.waitKey(0) 
cv2.destroyAllWindows()