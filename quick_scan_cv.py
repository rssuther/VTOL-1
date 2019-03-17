import cv2
import imutils
import json
import numpy as np
import os
from picamera.array import PiRGBArray
from picamera import PiCamera
from time import sleep
import string
import datetime

import sys
from digi.xbee.devices import RemoteXBeeDevice
from digi.xbee.models.address import XBee64BitAddress

import cv_img_metadata_class
import cv_img_poi

sys.path.append('..')
from VTOL.autonomy import setup_xbee

# Device MAC Address For Messaging
DEV_MAC_ADDR = "0013A200409BD79C"   # Needs to be set to correct device address
 
POI_ANALYSIS_LOC = './curr_poi_analysis/'
POI_ANALYSIS_FILE = './curr_poi_analysis/metadata.log'
FRAME_SAVE_LOC = './new_frames/'

IMAGE_META_DATA = []

# SET OPTIONS TO 1 TO ENABLE
SEND_INDV_POI = 0
SEND_ALL_POI = 0
SEND_IMAGE = 0
WRITE_META_DATA_FILE = 0


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
# @return PiCamera stream Object and raw capture array
# @error notify user of error
# @return upon error return error
###
def open_camera_stream():

    # Open camera stream (PiCamera) as video capture
    camera_cap = PiCamera()
    camera_cap.resolution = (640,480)
    camera_cap.framerate = 60
    camera_cap.awb_mode = 'auto'
    camera_cap.exposure_mode = 'auto'
    camera_cap.video_stabilization = True

    raw_capture = PiRGBArray(camera_cap)

    return (camera_cap, raw_capture)

###
# @param camera stream
# @error notify user of error
# @return upon error return error
###
def cloase_camera_stream(camera_cap):
    # Close Camera Stream
    camera_cap.close()

###
# Reads Frames in from PiCamera Durring VTOL Flight
# Performs POI Analysys
# @param video capture stream
# @error notify user of error
# @return upon error return error
###
def cv_img_read(camera_cap, raw_capture, autonomyToCV):

    # Loop for durration of flight
    frame_delay = 5
    frame_number = 1
    #frame_pool = len([file for file in os.listdir(FRAME_SAVE_LOC)])

    # Loop for durration of flight
    #while(AutonomyToCV.start == False)
    while(True):
        # Read frames from camera capture stream
        camera_cap.capture(raw_capture, format="bgr")
        frame = raw_capture.array

        # Read frames from path
        #frame_path = FRAME_SAVE_LOC + 'frame' + str(frame_number) + '.jpg'
        #frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)

        if frame_delay == 5:
            frame_path = FRAME_SAVE_LOC + 'frame' + str(frame_number) + '.jpg'
            cv2.imwrite(frame_path, frame)

            image_meta = cv_img_metadata_class.cv_img_meta().create(frame=frame, frame_path=frame_path, frame_number=frame_number)

            image_meta.set_img_gps_loc(autonomyToCV=autonomyToCV)
            
            if SEND_IMAGE:
                send_img_data(frame)

            (image_meta, poi_analysis) = cv_img_poi.contours_bounding(image_meta, autonomyToCV=autonomyToCV)

            frame_path = POI_ANALYSIS_LOC + 'frame' + str(frame_number) + '.jpg'
            cv2.imwrite(frame_path, poi_analysis)

            image_meta.set_img_poi_path(frame_path)

            IMAGE_META_DATA.append(image_meta)

            if SEND_INDV_POI:
                send_indv_poi_data(ent)

            frame_delay = 0
            frame_number += 1
            
        if autonomyToCV.stop == True:
            break
        
        frame_delay += 1
        raw_capture.truncate(0)

        #if frame_number == frame_pool-1:
        #    break
###
# Writes Data Collected on images to Metadata Log File
###
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

###
# Sends images from PiCamera to GCS as String
# Representations of the image array
# @param img: Image Captured from PiCamera
# @error notify user of error
###
def send_img_data(img):

    device = setup_xbee()

    try:
        remote_device = RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string(DEV_MAC_ADDR))

        MEAS_PREAMBLE = "Type: IMG, Size: %d"%len(img)

        device.send_data(remote_device, MEAS_PREAMBLE)

        MEAS_PAYLOAD = img.array
        device.send_data(remote_device, MEAS_PAYLOAD)

        device.send_data(remote_device, "END OF IMG")

        if remote_device is None:
            print("Invalid MAC Address for IMG Data")
            exit(1)
        
        print("Sending Data to %s" % ( remote_device.get_64bit_addr() ))

        # Determine number of bytes to be sent
        # Sent first message with type and size
        
        print("Success")
    
    finally:
        if device is not None and device.is_open():
            device.close()

###
# @param ent: POI Analysis Entry
# @error notify user of error
###
def send_indv_poi_data(ent):

    device = setup_xbee()

    try:
        remote_device = RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string(DEV_MAC_ADDR))

        if remote_device is None:
            print("Invalid MAC Address for POI Data")
            exit(1)
        
        print("Sending Data to %s" % ( remote_device.get_64bit_addr() ))

        MEAS_PREAMBLE = "Type: POI, Size: %d"%len(ent)

        device.send_data(remote_device, MEAS_PREAMBLE)
        
        POI_DATA = ent.get_img_poi()
        device.send_data(remote_device, POI_DATA)
        
        print("Success")
    
    finally:
        if device is not None and device.is_open():
            device.close()

###
# Sends All Collected POI Data At End Of Flight
# @error notify user of error
###
def send_all_poi_data():

    device = setup_xbee()

    try:
        remote_device = RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string(DEV_MAC_ADDR))

        if remote_device is None:
            print("Invalid MAC Address for POI Data")
            exit(1)
        
        print("Sending Data to %s" % ( remote_device.get_64bit_addr() ))

        MEAS_PREAMBLE = "Type: POI, Size: %d"%len(IMAGE_META_DATA)

        device.send_data(remote_device, MEAS_PREAMBLE)
        for ent in IMAGE_META_DATA:
            POI_DATA = ent.get_img_poi()
            device.send_data(remote_device, POI_DATA)
        
        print("Success")
    
    finally:
        if device is not None and device.is_open():
            device.close()

# FOR SUBSYSTEM DEBUGGING, SWITCH COMMENT STATE OF NEXT TWO LINES
# AND PASS autonomyToCV AS None
#def main():
def quick_scan_cv(configs, autonomyToCV):

    create_folders()
    (camera_cap, raw_capture) = open_camera_stream()

    START_TIME = datetime.datetime.now()

    cv_img_read(camera_cap, raw_capture, autonomyToCV)

    END_TIME = datetime.datetime.now()
    PROC_TIME = END_TIME - START_TIME

    close_camera_stream(camera_cap)

    print('POI Analysis Finished\n')
    
    print('Elapsed Time: '+str(PROC_TIME)+'\n')

    if WRITE_META_DATA_FILE:
        write_metadata_file()

    if SEND_ALL_POI:
        send_all_poi_data()
    exit()

if __name__ == '__main__':
	main()