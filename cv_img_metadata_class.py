import cv2
import json
from threading import Thread, Lock

class cv_img_meta():

    img_path = None
    img = None
    img_count = None
    img_kp = None
    img_desc = None
    img_stitch_pos = None
    img_poi_path = None
    img_poi = []
    img_poi_contours = None

    img_lat = None
    img_long = None
    img_alt = None
    img_north = None
    img_east = None

    def create():
        self.img = None
        return self

    def create(self, frame, frame_path, frame_number):
        self.img = frame
        self.img_path = frame_path
        self.img_count = frame_number
        self.id = frame_number

        return self

    def store_kps_desc(self, keypoints, descriptors):
        self.img_kp = keypoints
        self.img_desc = descriptors
    
    def get_kps_desc(self):
        return (self.img_kp, self.img_desc)

    def set_img_pos(self, x_pos = 0, y_pos = 0):
        self.img_stitch_pos = (x_pos, y_pos)
    
    def get_img_stitch_pos(self):
        return self.img_stitch_pos
    
    def set_img_poi(self, poi_pos):
        self.img_poi = poi_pos
    
    def add_img_poi(self, poi):
        self.img_poi.append(poi)

    def set_img_poi_path(self, poi_path):
        self.img_poi_path = poi_path

    def get_img_poi(self):
        return self.img_poi
    
    def get_img_poi_path(self):
        return self.img_poi_path
    
    def set_img_contours(self, poi_contour):
        self.img_poi_contours = poi_contour
    
    def get_img_contours(self):
        return self.img_poi_contours
    
    def get_image(self):
        return self.img
    
    def get_image_path(self):
        return self.img_path
    
    def get_image_number(self):
        return self.img_count

    def get_image_loc(self):
        return (self.img_lat, self.img_long, self.img_alt, self.img_north, self.img_east)

    def set_img_gps_loc(self, autonomyToCV=None, lat=0, long=0, alt=0, north=0, east=0): 
        if autonomyToCV != None:
            autonomyToCV.latMutex.acquire()
            self.img_lat = autonomyToCV.lat
            autonomyToCV.latMutex.release()            
        
            autonomyToCV.lonMutex.acquire()
            self.img_long = autonomyToCV.lon
            autonomyToCV.lonMutex.release()

            autonomyToCV.altMutex.acquire()
            self.img_alt = autonomyToCV.alt
            autonomyToCV.altMutex.release()

            autonomyToCV.northMutex.acquire()
            self.img_north = autonomyToCV.north
            autonomyToCV.northMutex.release()

            autonomyToCV.eastMutex.acquire()
            self.img_east = autonomyToCV.east
            autonomyToCV.eastMutex.release()

        else:
            self.img_lat = lat
            self.img_long = long
            self.img_alt = alt
            self.img_north = north
            self.img_east = east
    






