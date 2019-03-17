import cv2
import multiprocessing as mp
import time as t
import numpy as np
import cv_img_metadata_class
#import VTOL.quick_scan.py
from threading import Thread, Lock

def contours_bounding(image_meta, autonomyToCV):
	
	img_orig = image_meta.get_image()
	img = image_meta.get_image()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img,(5,5),0)
	ret, thresh = cv2.threshold(img , 195, 280, cv2.THRESH_BINARY_INV)
	#thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 1.8)
	image, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
	
	# Save Contours to Metadata

	image_meta.set_img_contours(contours)

	for c in contours:
		# find bounding box coordinates
		x,y,w,h = cv2.boundingRect(c)
		cv2.rectangle(img_orig, (x,y), (x+w, y+h), (0, 255, 0), 1)
	
		# find minimum area
		rect = cv2.minAreaRect(c)
		# calculate coordinates of the minimum area rectangle
		box = cv2.boxPoints(rect)
		# normalize coordinates to integers
		box = np.int0(box)
		# draw contours
		#cv2.drawContours(img, [box], 0, (0,0, 255), 1)
		# calculate center and radius of minimum enclosing circle
		(x,y),radius = cv2.minEnclosingCircle(c)
		# cast to integers
		center = (int(x),int(y))
		radius = int(radius)
		# draw the circle
		#img = cv2.circle(img,center,radius,(0,255,0),2)
		
		epsilon = 0.01 * cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)
	
		hull = cv2.convexHull(c)

	cv2.drawContours(img_orig, contours, -1, (255, 0, 0), 1)

	return (image_meta, img_orig)

def calculate_poi_loc(countour):
	# CHRISTIAN DO YOUR MAGIC HERE
	# Calculate the center points of all bounding boxes. and store them in a list like structure.
	# then write the data to the appropriot spot in image_meta
	# use the def add_img_poi(self, poi): fuction from the meta data class
	# return the image_meta object each time 
	pass
