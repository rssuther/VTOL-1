import cv2
import multiprocessing as mp
import time as t
import numpy as np
import cv_img_metadata_class
import VTOL.quick_scan.py
from threading import Thread, Lock
from conversions import *
from numpy import radians, sin, cos, tan, arctan, hypot, degrees, arctan2, sqrt, pi

##
# Analyses Images for POIs
# @param image_meta object
# @param autonomyToCV object
# @return image_meta, img_orig
##
def contours_bounding(image_meta, autonomyToCV):
	
	img_orig = image_meta.get_image()
	img = image_meta.get_image()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.GaussianBlur(img,(5,5),0)
	ret, thresh = cv2.threshold(img , 195, 280, cv2.THRESH_BINARY_INV)
	image, contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
	
	# Save Contours to Metadata Object
	image_meta.set_img_contours(contours)

	for c in contours:
		# Determine POI Location
		calculate_poi_loc(c, image_meta)

		# find bounding box coordinates
		x,y,w,h = cv2.boundingRect(c)
		cv2.rectangle(img_orig, (x,y), (x+w, y+h), (0, 255, 0), 1)
	
		# find minimum area
		rect = cv2.minAreaRect(c)
		# calculate coordinates of the minimum area rectangle
		box = cv2.boxPoints(rect)
		# normalize coordinates to integers
		box = np.int0(box)
		
		epsilon = 0.01 * cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, epsilon, True)
	
		hull = cv2.convexHull(c)

	cv2.drawContours(img_orig, contours, -1, (255, 0, 0), 1)

	return (image_meta, img_orig)
##
# Calculates the center points of all bounding boxes. and store them in the image metadata
# class. 
# @param countor: A single entry in the countours calculated
# @param image_meta: Metadata object
# @return image_meta object
##
def calculate_poi_loc(contour, image_meta):
	cx,cy,cw,ch = cv2.boundingRect(contour)
	(lat, lon, alt, n, e) = image_meta.get_image_loc()
	(cX, cY, cZ) = geodetic2ecef(lat, lon, alt) #Center of the image(in ECEF format)
	pcX = cx+(cw/2) #Center of the POI, in pixels, relative to top left of image
	pcY = cy+(ch/2) #Assuming the X,Y coordinates are from top left of image

	(x, y, chan) = image_meta.get_image().shape
	if(pcX > x/2): #If the contour is on the right side of the image
		if(pcY > y/2): #If the contour is on the top half of the image
			ccX = (x/2) + pcX
			ccY = (y/2) + pcY
			distY = y + ccY
		else: #Contour is on bottom half of image
			ccX = (x/2) + pcX
			ccY = (y/2) - pcY
			distY = y - ccY
		distX = x + ccX
	else: #Contour is on left half of image
		if(pcY > y/2): #If the contour is on the top half of the image
			ccX = (x/2) - pcX
			ccY = (y/2) + pcY
			distY = y + ccY
		else: #Contour is on bottom half of image
			ccX = (x/2) - pcX
			ccY = (y/2) - pcY
			distY = y - ccY
		distX = x - ccX

	#ccX and ccY are centers of the contour, relative to the entire image
	#distX and distY are the distances from the center of the image to the center of the contour
	#Convert the distance from center to meters and add relative to cX and cY(from GPS location)

	xRatio = distX / (x/2)
	yRatio = distY / (y/2)
	angX = xRatio * 26.75
	angY = yRatio * 41.41
	xMeters = arctan(radians(angX)) * distX
	yMeters = arctan(radians(angY)) * distY
	POIx = cX + xMeters
	POIy = cY + yMeters

	image_meta.add_img_poi((POIx, POIy))

	#53.50 horiz, 26.75 degrees in each direction
	#41.41 vert, 20.7 degrees in each direction

	return image_meta
