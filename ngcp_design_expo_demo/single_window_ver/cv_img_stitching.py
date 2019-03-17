import cv2
import imutils
import json
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

import cv_img_metadata_class

from opencv_imshow_fullscreen import show_demo



CURR_STITCHED_LOC = './curr_stitched_img/'
FRAME_SAVE_LOC = './new_frames/'
MIN_MATCH_COUNT = 4
MAX_PHOTO_COUNT = len([file for file in os.listdir(FRAME_SAVE_LOC)])   
recallkp = 1

# img_Stitcher object is run as a separate process to image frame capture

class ImgStitcher:

    FIRST_IMG = 1
    FRAME_COUNTER = 1
    STITCH_ITR = 1

    def __init__(self, img_ready_flag):
        # Determine version of OpenCV (v3.x)
        # Save to version variable
        self.isv3 = imutils.is_cv3()
        self.pointList = []
        self.featList = []


    def start(self, visualize, SHOW_VIS_OUTPUT):

        while (self.FRAME_COUNTER < MAX_PHOTO_COUNT):

            print('Stitch Iteration: '+str(self.STITCH_ITR))
            self.STITCH_ITR += 1

            if (self.FIRST_IMG):
                # First Frames
                self.FIRST_IMG = 0
                img1_path = FRAME_SAVE_LOC + 'frame' + str(self.FRAME_COUNTER) + '.jpg'
                self.FRAME_COUNTER += 1
                img2_path = FRAME_SAVE_LOC + 'frame' + str(self.FRAME_COUNTER) + '.jpg'
                self.FRAME_COUNTER += 1

            else:
                img1_path = CURR_STITCHED_LOC + 'current_stitched_area.jpg'
                img2_path = FRAME_SAVE_LOC + 'frame' + str(self.FRAME_COUNTER) + '.jpg'
                self.FRAME_COUNTER += 1

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            imgs = (img1, img2)

            if (SHOW_VIS_OUTPUT):
                if (self.FIRST_IMG): 
                    
                    show_demo('IMAGE STITCHING', img1, 1050)

                    #cv2.imshow('PICAM FRAME '+str(self.FRAME_COUNTER-1), img1)
                    
                    #cv2.namedWindow('IMAGE STITCHING', cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow('IMAGE STITCHING', 1920, 1080)
                    #cv2.moveWindow('IMAGE STITCHING', 0,0)

                    #cv2.waitKey(400)
                    #cv2.destroyWindow('IMAGE STITCHING'+str(self.FRAME_COUNTER-1))
                else:
                    #cv2.namedWindow('CURRENT STITCHED AREA', cv2.WINDOW_AUTOSIZE)
                    #cv2.resizeWindow('CURRENT STITCHED AREA', 1920, 1080)
                    #cv2.moveWindow('CURRENT STITCHED AREA', 120, 110)
                    

                    show_demo('IMAGE STITCHING', img1, 1050)

                    #cv2.imshow('IMAGE STITCHING', img1)
                    
                    #cv2.namedWindow('IMAGE STITCHING', cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow('IMAGE STITCHING', 1920, 1080)
                    #cv2.moveWindow('IMAGE STITCHING', 0,0)

                    #cv2.waitKey(400)
                    #cv2.destroyWindow('CURRENT STITCHED AREA')
       
                #cv2.namedWindow('FRAME '+str(self.FRAME_COUNTER), cv2.WINDOW_AUTOSIZE)
                #cv2.resizeWindow('FRAME '+str(self.FRAME_COUNTER), 1920, 1080)
                #cv2.moveWindow('FRAME '+str(self.FRAME_COUNTER), 120, 110)
                
                show_demo('IMAGE STITCHING', img2, 1050)

                #cv2.imshow('IMAGE STITCHING', img2)
                
                #cv2.namedWindow('IMAGE STITCHING', cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('IMAGE STITCHING', 1920, 1080)
                #cv2.moveWindow('IMAGE STITCHING', 0,0)

                #cv2.waitKey(400)
                #cv2.destroyWindow('FRAME '+str(self.FRAME_COUNTER))

            if visualize:
                (visualization, current_stitched_area) = self.stitch(imgs=imgs, visBool=True)
                cv2.imwrite(CURR_STITCHED_LOC + 'vis.jpg', visualization)

                if (SHOW_VIS_OUTPUT):
                    #cv2.namedWindow('KEYPOINT MATCHES', cv2.WINDOW_AUTOSIZE)
                    #cv2.resizeWindow('KEYPOINT MATCHES', 1920, 1080)
                    #cv2.moveWindow('KEYPOINT MATCHES', 120, 110)
                    
                    show_demo('KEYPOINT MATCHES', visualization, 1050)

                    #cv2.imshow('KEYPOINT MATCHES', visualization)
                    
                    #cv2.namedWindow('KEYPOINT MATCHES', cv2.WINDOW_NORMAL)
                    #cv2.resizeWindow('KEYPOINT MATCHES', 1920, 1080)
                    #cv2.moveWindow('KEYPOINT MATCHES', 0,0)

                    #cv2.waitKey(400)
                    #cv2.destroyWindow('KEYPOINT MATCHES')

            else:
                current_stitched_area = self.stitch(imgs=imgs, visBool=False)
            
            cv2.imwrite(CURR_STITCHED_LOC + 'current_stitched_area.jpg', current_stitched_area)
            
            if (SHOW_VIS_OUTPUT):

                #cv2.namedWindow('STITCHED AREA', cv2.WINDOW_AUTOSIZE)
                #cv2.resizeWindow('STITCHED AREA', 1920, 1080)
                #cv2.moveWindow('STITCHED AREA', 120, 110)
                

                show_demo('STITCHED AREA', current_stitched_area, 1050)

                #cv2.imshow('STITCHED AREA', current_stitched_area)
                
                #cv2.namedWindow('STITCHED AREA', cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('STITCHED AREA', 1920, 1080)
                #cv2.moveWindow('STITCHED AREA', 0,0)

                #cv2.waitKey(400)
                #cv2.destroyWindow('STITCHED AREA')


        cv2.destroyAllWindows()

    ###
    # @param_req self, array of two images
    # @param_opt self, array of two images, ratio, reprojThresh, visBool
    # @error notify user of error, handle
    # @return stitched image
    #
    # Detects keypoints in images and extracts local invariant descriptors
    # Matches features between images
    # Applys perspective warp and stitches images together
    # returns final stitched image
    ###
    def stitch(self, imgs, ratio=0.75, reprojThresh=4.0, visBool=False):
        # Detect Keypoints
        # Extract local invariant descriptors from keypoints

        print("Stitching...\n")

        (img_1, img_2) = imgs
        (kp_1, feat_1) = self.detect_describe(img_1)
        (kp_2, feat_2) = self.detect_describe(img_2)
        kp_3 = self.pointList
        feat_3 = self.featList
     
        # Match features between images
        if len(self.pointList) >0 and recallkp == 1:
            kp_1 = kp_3
            feat_1 = feat_3
        match_tuple = self.match_keypoints(kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh)
        match_tuple_keyPoint = match_tuple
        # If no matches are found return None
        if match_tuple is None:
            return None

        # Apply perspective warp and stitch images

        (height_img_1, width_img_1) = img_1.shape[:2]
        (height_img_2, width_img_2) = img_2.shape[:2]

        # Extract tuple
        (matches, homography, status, pts1, pts2) = match_tuple
        homography_x = homography
        val, homography_rev = cv2.invert(homography_x)
        
        newp = [[0,0],[0,0],[0,0],[0,0]]
        #0,0
        newp[0][0] = 0*homography_rev[0][0]+0*homography_rev[0][1]+1*homography_rev[0][2]
        newp[0][1] = 0*homography_rev[1][0]+0*homography_rev[1][1]+1*homography_rev[1][2] 
        #0,height_img_2
        newp[1][0] = 0*homography_rev[0][0]+height_img_2*homography_rev[0][1]+1*homography_rev[0][2]
        newp[1][1] = 0*homography_rev[1][0]+height_img_2*homography_rev[1][1]+1*homography_rev[1][2] 
        #width_img_2, 0
        newp[2][0] = width_img_2*homography_rev[0][0]+0*homography_rev[0][1]+1*homography_rev[0][2]
        newp[2][1] = width_img_2*homography_rev[1][0]+0*homography_rev[1][1]+1*homography_rev[1][2] 
        #width_img_2, height_img_2
        newp[3][0] = width_img_2*homography_rev[0][0]+height_img_2*homography_rev[0][1]+1*homography_rev[0][2]
        newp[3][1] = width_img_2*homography_rev[1][0]+height_img_2*homography_rev[1][1]+1*homography_rev[1][2] 
    
        maxx = 0
        maxy = 0
        minx = 0
        miny = 0
        readjust = 0
        for i in range (0,4):
            #print(newp[i][0], newp[i][1])
            if maxx<newp[i][0]:
                maxx =int(newp[i][0])
            if maxy<newp[i][1]:
                maxy = int(newp[i][1])
            if newp[i][0] <-1:
                readjust = 1
            if newp[i][1] <-1:
                readjust = 1
            if minx > newp[i][0]:
                minx =int(newp[i][0])
            if miny > newp[i][1]:
                miny = int(newp[i][1])
                
        if readjust :
            readjust = 1
            x_adjust = minx*-1
            y_adjust = miny*-1
            imgborder = cv2.copyMakeBorder(img_1, y_adjust, 0, x_adjust, 0, cv2.BORDER_CONSTANT, None, 0)
            (height_img_1, width_img_1) = imgborder.shape[:2]
            
            for i in range(0, len(kp_1)):
                kp_1[i][0] = kp_1[i][0] + x_adjust
                kp_1[i][1] = kp_1[i][1] + y_adjust
            
            match_tuple = self.match_keypoints(kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh)
            match_tuple_keyPoint = match_tuple
            # If no matches are found return None
            if match_tuple is None:
                return None
    
            # Extract tuple
            (matches, homography, status, pts1, pts2) = match_tuple
            
            # Perform warpPerspective on img_2, Apply Translation
            newimage_x = width_img_1
            newimage_y = height_img_1
            required_x = maxx
            required_y = maxy
            if newimage_x > required_x:
                actualimage_x = newimage_x
            else:
                actualimage_x = required_x
            if newimage_y > required_y:
                actualimage_y = newimage_y
            else:
                actualimage_y = required_y
            stitch_img = cv2.warpPerspective(img_2, homography, (actualimage_x, actualimage_y), flags=cv2.WARP_INVERSE_MAP)
            
            # Fill Area with Current Stitched Area 
            output_stitch_img = np.zeros((actualimage_y,actualimage_x,3), dtype="uint8")

            output_stitch_img[0:height_img_1, 0:width_img_1] = imgborder
            
            
            for i in range(0, actualimage_y):
                for j in range(0, actualimage_x):
                    if output_stitch_img[i][j][0] == 0 or output_stitch_img[i][j][1] == 0 or output_stitch_img[i][j][2] == 0:
                        output_stitch_img[i][j] = stitch_img[i][j]

            
            
        else:   
            # Perform warpPerspective on img_2, Apply Translation
            newimage_x = width_img_1
            newimage_y = height_img_1
            required_x = maxx
            required_y = maxy
            if newimage_x > required_x:
                actualimage_x = newimage_x
            else:
                actualimage_x = required_x
            if newimage_y > required_y:
                actualimage_y = newimage_y
            else:
                actualimage_y = required_y
            stitch_img = cv2.warpPerspective(img_2, homography, (actualimage_x, actualimage_y), flags=cv2.WARP_INVERSE_MAP)
            
             # Fill Area with Current Stitched Area 
            output_stitch_img = np.zeros((actualimage_y,actualimage_x,3), dtype="uint8")

            output_stitch_img[0:height_img_1, 0:width_img_1] = img_1
        
            
            for i in range(0, actualimage_y):
                for j in range(0, actualimage_x):
                    if output_stitch_img[i][j][0] == 0 or output_stitch_img[i][j][1] == 0 or output_stitch_img[i][j][2] == 0:
                        output_stitch_img[i][j] = stitch_img[i][j]
            

        if readjust:
            
            val, homography_rev = cv2.invert(homography)
            new_kp = cv2.perspectiveTransform(np.array([kp_2]), homography_rev)
            
            new_kp2 = np.append(new_kp[0], kp_1, axis = 0)
            total_feat = np.append(feat_2, feat_1, axis = 0)
            count = 0
            for (x,(y,z)) in enumerate (matches): 
                new_kp2[y][0] = -1
                count = count+1
            #NOT WOKING YET
            count = 0
            iterx = 0
            while iterx < len(new_kp2):
                if new_kp2[iterx][0] == -1:
                    new_kp2 = np.delete(new_kp2, iterx, axis = 0)
                    total_feat = np.delete(total_feat, iterx, axis = 0)
                    count = count+1
                    
                    if count>= len(matches):
                        break
                else:
                    iterx = iterx + 1
            self.pointList = new_kp2
            self.featList = total_feat
                    
        else:
            #calculate homograaphy for keypoint
    
            val, homography = cv2.invert(homography_x)
            new_kp = cv2.perspectiveTransform(np.array([kp_2]), homography)
            
            
            new_kp2 = np.append(new_kp[0], kp_1, axis = 0)
            total_feat = np.append(feat_2, feat_1, axis = 0)
            count = 0
            for (x,(y,z)) in enumerate (matches): 
                new_kp2[z][0] = -1
                count = count+1
            count = 0
            iterx = 0
            while iterx < len(new_kp2):
                if new_kp2[iterx][0] == -1:
                    new_kp2 = np.delete(new_kp2, iterx, axis = 0)
                    total_feat = np.delete(total_feat, iterx, axis = 0)
                    count = count+1
                    
                    if count>= len(matches):
                        break
                else:
                    iterx = iterx + 1
            
            self.pointList = new_kp2
            self.featList = total_feat

        # Check visualization boolean for keypoints
        if visBool:
            if readjust:
                ouptut_matches_img = self.draw_matches(img_1, img_2, kp_1, kp_2, matches, status)
            else:
                ouptut_matches_img = self.draw_matches(img_1, img_2, kp_1, kp_2, matches, status)

            #return tuple of stitched image and visualization
            return(ouptut_matches_img, output_stitch_img)
        
        #return stitched image
        return output_stitch_img
    
    ###
    # @param self, image
    # @error notify user of error, handle
    # @return tuple(numpy array of kp, features)
    # 
    # detect_describe() converts the provided image to a greyscale
    # image format for manipulation. 
    # Detects and computes the image features.
    # Converts Keypoint objects to numpy array
    # 
    # Returns tuple of keypoints and features
    ###
    def detect_describe(self, img):
        # Convert image to greyscale
        
        print("Detect and Describe Keypoints...\n")

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Check Opencv version param
        if self.isv3:
            # Detect and extract features from image
            desc = cv2.xfeatures2d.SIFT_create()
            (kps, feat) = desc.detectAndCompute(img, None)

        else:  
            # detect keypoints in image
            detect = cv2.FeatureDetector_create("SIFT")
            kps = detect.detect(gray_img)

            #extract features from image
            extract = cv2.DescriptorExtractor_create("SIFT")
            (kps, feat) = extract.compute(gray, kps)
        
        # Convert keypoints from KP objects to Numpy arrays (float32 type)
        kps = np.float32([pt.pt for pt in kps])

        #return tuple of kp and features
        return (kps, feat)

    ###
    # @param kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh
    # @error notify user of error, handle
    # @return tuple(matches along with homography matrix and status) or None
    # 
    # match_keypoints() computes raw matches
    # performs Lowe's ratio test on matches
    # computes homography
    # returns tuple of matches, homography matrix and status
    ###
    def match_keypoints(self, kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh):
        # Compute raw matches
        
        print("Compute Keypoint Matches...\n")

        match = cv2.DescriptorMatcher_create("BruteForce")
        matches_raw = match.knnMatch(feat_1, feat_2, 2)

        # Initialize matches array
        matches = []

        # Loop over each match in raw matches
        for m in matches_raw:
            # Use Lowe's ratio test to ensure distance is within certain ratio
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Compute homography (requiers at least 4 matches)
        if len(matches) > MIN_MATCH_COUNT:

            # Construct two sets of points
            pts1 = np.float32([kp_1[i] for (_, i) in matches])

            pts2 = np.float32([kp_2[i] for (i, _) in matches])

            # Compute homography between sets
            (homography, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, reprojThresh)

            #return matches along with homography matrix and status
            return (matches, homography, status, pts1, pts2)

        # if not at least 4
        # return None
        else:
            print ("Not enough matches are found for homography - %d/%d" % (len(matches),MIN_MATCH_COUNT))
            return None

    ###
    # @param self, img_1, img_2, kp_1, kp_2, matches, ststus)
    # @error notify user of error, handle
    # @return Matches Visualization Image
    #
    # draw_matches() draws a line between all matches
    # Returns final visualization
    ###
    def draw_matches(self, img_1, img_2, kp_1, kp_2, matches, status):
        # Initialize output visualization image
        
        print("Draw Keypoint Matches Visualization...\n")

        (height_img_1, width_img_1) = img_1.shape[:2]
        (height_img_2, width_img_2) = img_2.shape[:2]
        output_matches_img = np.zeros((max(height_img_1, height_img_2), width_img_1+ width_img_2, 3), dtype="uint8")
        output_matches_img[0:height_img_1, 0:width_img_1] = img_1
        output_matches_img[0:height_img_2, width_img_1:] = img_2
        
        # Loop over each match in matches
        for ((train_index, query_index), sucsess) in zip(matches, status):
            #Only process match if kp was sucessful
            if sucsess == 1:
                # Draw Match
                point_1 = (int(kp_1[query_index][0]), int(kp_1[query_index][1]))
                point_2 = (int(kp_2[train_index][0]) + width_img_1, int(kp_2[train_index][1]))
                cv2.line(output_matches_img, point_1, point_2, (0, 255, 0), 1)

        # Return Visualization
        return output_matches_img

def create_folders():
    # If folders dont already exist, create folders
    if not os.path.exists(FRAME_SAVE_LOC):
        os.makedirs(FRAME_SAVE_LOC)
    if not os.path.exists(CURR_STITCHED_LOC):
        os.makedirs(CURR_STITCHED_LOC)

def img_stitching_demo(show_vis=0):
#def main():
    create_folders()

    START_TIME = datetime.datetime.now()

    stitcher = ImgStitcher(1)
    stitcher.start(visualize=True, SHOW_VIS_OUTPUT=show_vis)
    
    END_TIME = datetime.datetime.now()
    STITCH_TIME = END_TIME - START_TIME
    print('Stitching Process Finished\n')
    print('Frames Stitched: '+str(stitcher.FRAME_COUNTER-1)+'\n')
    print('Elapsed Time: '+str(STITCH_TIME)+'\n')

    cv2.destroyAllWindows()

    exit(0)

#if __name__ == '__main__':
#    main()

    
