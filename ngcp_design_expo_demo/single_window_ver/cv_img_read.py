import cv2
import imutils
import json
import numpy as np
import os
#from picamera.array import PiRGBArray
#from picamera import PiCamera
from time import sleep
import string
import datetime

import sys
#from digi.xbee.devices import RemoteXBeeDevice
#from digi.xbee.models.address import XBee64BitAddress

import matplotlib
import matplotlib.pyplot as plt

import cv_img_metadata_class
import cv_img_poi

from opencv_imshow_fullscreen import show_demo

#sys.path.append('..')
#from VTOL.autonomy import setup_xbee

MAC_ADDR = "0013A200409BD79C"
 
POI_ANALYSIS_LOC = './curr_poi_analysis/'
POI_ANALYSIS_FILE = './curr_poi_analysis/metadata.log'
FRAME_SAVE_LOC = './new_frames/'

IMAGE_META_DATA = []

###
# @error notify user of error
# @return upon error return error
###
def create_folders():
    # If folders dont already exist, create folders
    if not os.path.exists(POI_ANALYSIS_LOC):
        os.makedirs(POI_ANALYSIS_LOC)

    if not os.path.exists(FRAME_SAVE_LOC):
        os.makedirs(FRAME_SAVE_LOC)

###
# @return Camera stream object
# @error notify user of error
# @return upon error return error
###
def open_camera_stream():
    # Open camera stream (0) as video capture
    #camera_cap = cv2.VideoCapture(0)

    # Open camera stream (PiCamera) as video capture
    camera_cap = PiCamera()
    camera_cap.resolution = (640,480)
    camera_cap.framerate = 60
    rawCapture = PiRGBArray(camera_cap, size=(640, 480))

    camera_cap.set(5, 60)
    camera_cap.set(3, 320)
    camera_cap.set(4, 420)

    return camera_cap

###
# @param video capture stream
# @error notify user of error
# @return upon error return error
###
#def cv_img_read(camera_cap, AutonomyToCV):
def cv_img_read(camera_cap, autonomyToCV, SHOW_VIS_OUTPUT=1):

    # Loop for durration of flight
    frame_delay = 10
    frame_number = 1
    frame_pool = len([file for file in os.listdir(FRAME_SAVE_LOC)])

    cv2.namedWindow('POI ANALYSIS', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('POI ANALYSIS', 1920, 1080)
    cv2.moveWindow('POI ANALYSIS', 0,0)


    # Loop for durration of flight
    #while(AutonomyToCV.start == False)
    while(frame_number != frame_pool-1):
        # Read frames from camera capture stream
        #ret, frame = camera_cap.read()
        #rawCapture = PiRGBArray(camera_cap, size=(640, 480))
        #frame = cv2.imread(frame, cv2.IMREAD_UNCHANGED)

        # Read frames from path
        frame_path = FRAME_SAVE_LOC + 'frame' + str(frame_number) + '.jpg'
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if frame_delay == 10:
            #frame_path = FRAME_SAVE_LOC + 'frame' + str(frame_number) + '.jpg'
            #cv2.imwrite(frame_path, frame)
            
            print("Analyzing Frame %d...\n" % frame_number)

            if (SHOW_VIS_OUTPUT):
                #cv2.namedWindow('PICAM FRAME '+str(frame_number), cv2.WINDOW_AUTOSIZE)
                #cv2.moveWindow('PICAM FRAME '+str(frame_number), 120, 110)
                #cv2.resizeWindow('PICAM FRAME '+str(frame_number), 1920, 1080)
                cv2.imshow('PICAM FRAME', frame)

                cv2.namedWindow('PICAM FRAME', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('PICAM FRAME', 1920, 1080)
                cv2.moveWindow('PICAM FRAME', 0,0)

                cv2.waitKey(400)
                #cv2.destroyWindow('PICAM FRAME '+str(frame_number))

            image_meta = cv_img_metadata_class.cv_img_meta().create(frame=frame, frame_path=frame_path, frame_number=frame_number)

            image_meta.set_img_gps_loc(autonomyToCV=None, lat=0, long=0, alt=0, north=0, east=0)

            (image_meta, poi_analysis) = cv_img_poi.contours_bounding(image_meta, autonomyToCV=None)
            frame_path = POI_ANALYSIS_LOC + 'frame' + str(frame_number) + '.jpg'
            cv2.imwrite(frame_path, poi_analysis)

            if (SHOW_VIS_OUTPUT):
                #cv2.namedWindow('POI ANALYSIS FRAME '+str(frame_number), cv2.WINDOW_AUTOSIZE)
                #cv2.resizeWindow('POI ANALYSIS FRAME '+str(frame_number), 1920, 1080)
                #cv2.moveWindow('POI ANALYSIS FRAME '+str(frame_number), 120, 110)
                cv2.imshow('POI ANALYSIS FRAME', poi_analysis)
                
                cv2.namedWindow('POI ANALYSIS FRAME', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('POI ANALYSIS FRAME', 1920, 1080)
                cv2.moveWindow('POI ANALYSIS FRAME', 0,0)

                cv2.waitKey(400)
                #cv2.destroyWindow('POI ANALYSIS FRAME '+str(frame_number))

            image_meta.set_img_poi_path(frame_path)

            IMAGE_META_DATA.append(image_meta)
            
            frame_delay = 10
            frame_number += 1
            
        #if autonomyToCV.stop == True:
        #    break
        
        #frame_delay += 1


        cv2.destroyAllWindows()

def write_metadata_file():

    metafile = open(POI_ANALYSIS_FILE, mode='w+')

    metafile.write("METADATA:\n\n")
        
    for ent in IMAGE_META_DATA:
        metafile.write("## START METADATA SEQUENCE "+str(ent.get_image_number())+"\n")
        metafile.write("IMAGE PATH: "+str(ent.get_image_path())+"\n")
        metafile.write("IMAGE PATH (POI): "+str(ent.get_img_poi_path())+"\n")
        (lat, long, alt, north, east) = ent.get_image_loc()
        metafile.write("LOC: lat:"+str(lat)+" long:"+str(long)+" alt:"+str(alt)+" north:"+str(north)+" east:"+str(east)+"\n")
        (kp, desc) = ent.get_kps_desc()
        metafile.write("KEYPOINTS: "+str(kp)+"\n")
        metafile.write("DESCRIPTORS: "+str(desc)+"\n")
        metafile.write("STITCH POSITION: "+str(ent.get_img_stitch_pos())+"\n")
        metafile.write("CONTOURS:\n"+str(ent.get_img_contours())+"\n")
        metafile.write("\nPOI: "+str(ent.get_img_poi())+"\n")
        metafile.write("## END METADATA SEQUENCE "+str(ent.get_image_number())+"\n\n")

    metafile.close()

def send_poi_data():

    device = setup_xbee()

    try:
        remote_device = RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string(MAC_ADDR))

        if remote_device is None:
            print("Invalid MAC Address for POI Data")
            exit(1)
        
        print("Sending Data to %s" % ( remote_device.get_64bit_addr() ))

        for ent in IMAGE_META_DATA:
            POI_DATA = ent.get_img_poi()
            device.send_data(remote_device, POI_DATA)
        
        print("Success")
    
    finally:
        if device is not None and device.is_open():
            device.close()


def img_poi_detection_demo(show_vis=0):

    create_folders()
    #camera_cap = open_camera_stream()

    START_TIME = datetime.datetime.now()

    cv_img_read(None, None, SHOW_VIS_OUTPUT=show_vis)

    END_TIME = datetime.datetime.now()
    PROC_TIME = END_TIME - START_TIME

    #close_camera_stream(camera_cap)

    print('POI Analysis Finished\n')
    
    print('Elapsed Time: '+str(PROC_TIME)+'\n')

    #write_metadata_file()

    #send_poi_data()
    exit(0)
