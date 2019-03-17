import cv2
import imutils
import json
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import datetime
import cv_img_metadata_class

CURR_STITCHED_LOC = './curr_stitched_img/'
FRAME_SAVE_LOC = './new_frames/'
MIN_MATCH_COUNT = 8
MAX_PHOTO_COUNT = len([file for file in os.listdir(FRAME_SAVE_LOC)])   
recallkp = 1
USE_FLANNE = 1 
USE_ORB = 0   #Not working
USE_PREV_KP = 1
SHOW_KP_AT_END = 0
NO_IMAGE_KP_ONLY = 0

# img_Stitcher object is run as a separate process to image frame capture

class ImgStitcher:

    CURR_STITCHED_LOC = './curr_stitched_img/'
    FRAME_SAVE_LOC = './new_frames/'
    FIRST_IMG = 1
    FRAME_COUNTER = 1
    STITCH_ITR = 1

    def __init__(self, img_ready_flag):
        # Determine version of OpenCV (v3.x)
        # Save to version variable
        self.isv3 = imutils.is_cv3()
        self.total_kpList = []
        self.featList = []
        self.prev_kpList = []
        self.prev_featList = []

        #loop till end of flight
            # wait till image data is available (img_ready_f)
            # clear flag or respond with message

    def start(self, SHOW_VIS_OUTPUT):
        while (self.FRAME_COUNTER < MAX_PHOTO_COUNT):

            print('Stitch Iteration: '+str(self.STITCH_ITR))
            self.STITCH_ITR += 1

            if (self.FIRST_IMG):
                # First Frame
                self.FIRST_IMG = 0
                img1_path = self.FRAME_SAVE_LOC + 'frame' + str(self.FRAME_COUNTER) + '.jpg'
                self.FRAME_COUNTER += 1
                img2_path = self.FRAME_SAVE_LOC + 'frame' + str(self.FRAME_COUNTER) + '.jpg'
                self.FRAME_COUNTER += 1

            else:
                img1_path = self.CURR_STITCHED_LOC + 'current_stitched_area.jpg'
                img2_path = self.FRAME_SAVE_LOC + 'frame' + str(self.FRAME_COUNTER) + '.jpg'
                self.FRAME_COUNTER += 1

            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            imgs = (img1, img2)

            if (SHOW_VIS_OUTPUT):
                if (self.FIRST_IMG): 
                    cv2.namedWindow('PICAM FRAME '+str(self.FRAME_COUNTER-1), cv2.WINDOW_AUTOSIZE)
                    #cv2.resizeWindow('PICAM FRAME '+str(self.FRAME_COUNTER-1), 1920, 1080)
                    cv2.moveWindow('PICAM FRAME '+str(self.FRAME_COUNTER-1), 120, 110)
                    cv2.imshow('PICAM FRAME '+str(self.FRAME_COUNTER-1), img1)
                    cv2.waitKey(400)
                    cv2.destroyWindow('PICAM FRAME '+str(self.FRAME_COUNTER-1))
                else:
                    cv2.namedWindow('CURRENT STITCHED AREA', cv2.WINDOW_AUTOSIZE)
                    #cv2.resizeWindow('CURRENT STITCHED AREA', 1920, 1080)
                    cv2.moveWindow('CURRENT STITCHED AREA', 120, 110)
                    cv2.imshow('CURRENT STITCHED AREA', img1)
                    cv2.waitKey(400)
                    cv2.destroyWindow('CURRENT STITCHED AREA')
       
                cv2.namedWindow('FRAME '+str(self.FRAME_COUNTER), cv2.WINDOW_AUTOSIZE)
                #cv2.resizeWindow('FRAME '+str(self.FRAME_COUNTER), 1920, 1080)
                cv2.moveWindow('FRAME '+str(self.FRAME_COUNTER), 120, 110)
                cv2.imshow('FRAME '+str(self.FRAME_COUNTER), img2)
                cv2.waitKey(400)
                cv2.destroyWindow('FRAME '+str(self.FRAME_COUNTER))

            if NO_IMAGE_KP_ONLY:
                visualize = False
            else:
                visualize = True
            if visualize:
                (visualization, current_stitched_area) = self.stitch(imgs=imgs, visBool=visualize)
                cv2.imwrite(self.CURR_STITCHED_LOC + 'vis.jpg', visualization)

                if (SHOW_VIS_OUTPUT):
                    cv2.namedWindow('KEYPOINT MATCHES', cv2.WINDOW_AUTOSIZE)
                    #cv2.resizeWindow('KEYPOINT MATCHES', 1920, 1080)
                    cv2.moveWindow('KEYPOINT MATCHES', 120, 110)
                    cv2.imshow('KEYPOINT MATCHES', visualization)
                    cv2.waitKey(400)
                    cv2.destroyWindow('KEYPOINT MATCHES')

            else:
                current_stitched_area = self.stitch(imgs=imgs, visBool=False)

            if NO_IMAGE_KP_ONLY == 0:

                cv2.imwrite(self.CURR_STITCHED_LOC + 'current_stitched_area.jpg', current_stitched_area)

                if (SHOW_VIS_OUTPUT):

                    cv2.namedWindow('STITCHED AREA', cv2.WINDOW_AUTOSIZE)
                    #cv2.resizeWindow('STITCHED AREA', 1920, 1080)
                    cv2.moveWindow('STITCHED AREA', 120, 110)
                    cv2.imshow('STITCHED AREA', current_stitched_area)
                    cv2.waitKey(400)
                    cv2.destroyWindow('STITCHED AREA')
            
        if SHOW_KP_AT_END:
            for i in range (0, len(self.total_kpList)):
                plt.plot(self.total_kpList[i][0], self.total_kpList[i][1], 'bo')
            plt.axis([-200, 700, -200, 700])
            plt.ylabel('kp_2 new')
            plt.show()

        ######
            #start image stitch process
                #if first images:
                    #read new frames from folder "new_frames":"frame1.jpg","frame2.jpg"
                    # after frames are collected, remove them from folder (folder will only ever contain two frames named "frame1.jpg", "frame2.jpg")
                #else (get new frame and current stitched image)
                    # Read current stitch from folder "curr_stitched_img":"curr_stitched_area.jpg"
                    #read new frame from folder "new_frames":"frame1.jpg"
                # call stitch() function
                # write result to "curr_stitched_img":"curr_stitched_area.jpg"
            # Check xbee messages for flight concluded

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
        (img_1, img_2) = imgs
        (kp_2, feat_2) = self.detect_describe(img_2)

        
        
        #Recallkp = 1: use kp from previous frame for matching.
        #Recallkp = 0: read in current stitched image.
        if recallkp == 0:
            (kp_1, feat_1) = self.detect_describe(img_1)
        else:
            if len(self.prev_kpList) >0 and USE_PREV_KP == 1:
                #USE_PREV_KP = 1: use kp from previous one frame
                kp_1 = self.prev_kpList
                feat_1 = self.prev_featList 
            elif len(self.total_kpList) >0 :
                #USE_PREV_KP = 0: use kp from entire stored list
                kp_1 = self.total_kpList
                feat_1 = self.featList
            else:
                #read in stitched image
                (kp_1, feat_1) = self.detect_describe(img_1)
                
                
        # Match features between images
        match_tuple = self.match_keypoints(kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh)
        
        # If no matches are found, use full keypoint list
        if match_tuple == False:
            kp_1 = self.total_kpList
            feat_1 = self.featList
            match_tuple = self.match_keypoints(kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh)
        if match_tuple == False:
            # If still no matches are found, read in stitched image
            (kp_1, feat_1) = self.detect_describe(img_1)
            match_tuple = self.match_keypoints(kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh)

        
        match_tuple_keyPoint = match_tuple
        if match_tuple is None:
            return None

        # Apply perspective warp and stitch images
        (height_img_1, width_img_1) = img_1.shape[:2]
        (height_img_2, width_img_2) = img_2.shape[:2]
        
        #Showing current matching image
        '''
        if NO_IMAGE_KP_ONLY == 0:
            cv2.imshow('img_1', img_1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imshow('img_2', img_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        '''

        # Extract tuple
        (matches, homography, status, pts1, pts2) = match_tuple
        homography_x = homography
        val, homography_rev = cv2.invert(homography_x)
        
        #Calculate image location after homography
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
        minx = sys.maxsize
        miny = sys.maxsize
        readjust = 0
        readjustx = 0
        readjusty = 0
        for i in range (0,4):
            if maxx<newp[i][0]:
                maxx =int(newp[i][0])
            if maxy<newp[i][1]:
                maxy = int(newp[i][1])
            if newp[i][0] <-1:
                readjust = 1
                readjustx = 1
            if newp[i][1] <-1:
                readjust = 1
                readjusty = 1
            if minx > newp[i][0]:
                minx =int(newp[i][0])
            if miny > newp[i][1]:
                miny = int(newp[i][1])
                
        if readjust :
            x_adjust = 0
            y_adjust = 0
            
            #Calculate re adjust distance
            if readjustx:
                x_adjust = minx*-1
            if readjusty:
                y_adjust = miny*-1
            imgborder = cv2.copyMakeBorder(img_1, y_adjust, 0, x_adjust, 0, cv2.BORDER_CONSTANT, None, 0)
            (height_img_1, width_img_1) = imgborder.shape[:2]

            newimage_x = width_img_1
            newimage_y = height_img_1
            
            required_x = maxx + x_adjust
            required_y = maxy + y_adjust
            if newimage_x > required_x:
                actualimage_x = newimage_x
            else:
                actualimage_x = required_x
            if newimage_y > required_y:
                actualimage_y = newimage_y
            else:
                actualimage_y = required_y
                
            #Re calculate the homography again based on the readjust distance
            if USE_ORB:
                for i in range(0, len(kp_1)):
                    kp_1[i].pt = (kp_1[i].pt[0] + x_adjust, kp_1[i].pt[1] + y_adjust)
            else:
                for i in range(0, len(kp_1)):
                    kp_1[i][0] = kp_1[i][0] + x_adjust
                    kp_1[i][1] = kp_1[i][1] + y_adjust
            match_tuple = self.match_keypoints(kp_1, kp_2, feat_1, feat_2, ratio, reprojThresh)
            
            # If no matches are found return None
            if match_tuple is None:
                return None
    
            # Extract tuple
            (matches, homography, status, pts1, pts2) = match_tuple
                
            #Perform image stitching
            # Perform warpPerspective on img_2, Apply Translation 
            stitch_img = cv2.warpPerspective(img_2, homography, (actualimage_x, actualimage_y), flags=cv2.WARP_INVERSE_MAP)
            
            '''
            if NO_IMAGE_KP_ONLY == 0:
                cv2.imshow('warpPerspective', stitch_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            '''
            
            # Fill Area with Current Stitched Area 
            output_stitch_img = np.zeros((actualimage_y,actualimage_x,3), dtype="uint8")

            output_stitch_img[0:height_img_1, 0:width_img_1] = imgborder
            
            '''
            if NO_IMAGE_KP_ONLY == 0:
                cv2.imshow('output_stitch_img', output_stitch_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                for i in range(miny+y_adjust, maxy+y_adjust-1):
                    for j in range(minx + x_adjust, maxx + x_adjust-1):
                        if output_stitch_img[i][j][0] == 0 or output_stitch_img[i][j][1] == 0 or  output_stitch_img[i][j][2] == 0:
                            output_stitch_img[i][j] = stitch_img[i][j]
                cv2.imshow('warpPerspective -1', output_stitch_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            '''
        else:   
            if NO_IMAGE_KP_ONLY == 0:
                #Perform image stitching
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
                
                '''
                cv2.imshow('warpPerspective', stitch_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''
            
                 # Fill Area with Current Stitched Area 
                output_stitch_img = np.zeros((actualimage_y,actualimage_x,3), dtype="uint8")
                 
                output_stitch_img[0:height_img_1, 0:width_img_1] = img_1
                
                '''
                cv2.imshow('output_stitch_img', output_stitch_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''

                for i in range(miny, maxy):
                    for j in range(minx, maxx):
                        if output_stitch_img[i][j][0] == 0 or output_stitch_img[i][j][1] == 0 or output_stitch_img[i][j][2] == 0:
                            output_stitch_img[i][j] = stitch_img[i][j]
                
                '''
                cv2.imshow('warpPerspective -1', output_stitch_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                '''

        #Perform transformation on kp, and store it.
        if readjust:
            
            val, homography_rev = cv2.invert(homography)
            new_kp = cv2.perspectiveTransform(np.array([kp_2]), homography_rev)
            
            if USE_PREV_KP:
                self.prev_kpList = new_kp[0]
                self.prev_featList = feat_2
            
            if len(self.total_kpList) == 0:
                new_kp2 = new_kp[0]
                total_feat = feat_2
            else:
                new_kp2 = np.append(new_kp[0], self.total_kpList, axis = 0)
                total_feat = np.append(feat_2, self.featList, axis = 0)   
            

            self.total_kpList = new_kp2
            self.featList = total_feat
            #print('total kp len:', len(self.total_kpList))
        else:
            #calculate homography for keypoint
            val, homography = cv2.invert(homography_x)
            new_kp = cv2.perspectiveTransform(np.array([kp_2]), homography)
            dummy = new_kp[0]
            
            if USE_PREV_KP:
                self.prev_kpList = dummy
                self.prev_featList = feat_2
            if len(self.total_kpList) == 0:
                new_kp2 = new_kp[0]
                total_feat = feat_2
            else:
                new_kp2 = np.append(new_kp[0], self.total_kpList, axis = 0)
                total_feat = np.append(feat_2, self.featList, axis = 0)   

            self.total_kpList = new_kp2
            self.featList = total_feat
            #print('total kp len:', len(self.total_kpList))
        
        # Check visualization boolean for keypoints
        if visBool:
            if readjust:
                ouptut_matches_img = self.draw_matches(img_1, img_2, kp_1, kp_2, matches, status)
            else:
                ouptut_matches_img = self.draw_matches(img_1, img_2, kp_1, kp_2, matches, status)

            #return tuple of stitched image and visualization
            if NO_IMAGE_KP_ONLY :
                return None
            else:
                return(ouptut_matches_img, output_stitch_img)
        
        #return stitched image
        if NO_IMAGE_KP_ONLY:
            return None
        else:
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
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Check Opencv version param
        if self.isv3:
            # Detect and extract features from image
            # Use cv2.xfeatures2d.SIFT_create()
            desc = cv2.xfeatures2d.SIFT_create()
            # Use .detectAndCompute(gimg, None) method from xfeatures2d
            (kps, feat) = desc.detectAndCompute(img, None)
# =============================================================================
#             orb = cv2.ORB_create()
#             kporb = orb.detect(img,None)
#             kporb, des = orb.compute(img, kporb)
# =============================================================================

        else:  
            # detect keypoints in image
            # Use cv2.FeatureDetector_create("SIFT")]
            detect = cv2.FeatureDetector_create("SIFT")
            # Use .detect methood of FeatureDetector on gimg
            kps = detect.detect(gray_img)

            #extract features from image
            # Use cv2.DescriptorExtractor_create("SIFT")
            extract = cv2.DescriptorExtractor_create("SIFT")
            # Use .compute methood of DescriptorExtractor on gimg
            (kps, feat) = extract.compute(gray, kps)
        
        # Convert keypoints from KP objects to Numpy arrays (float32 type)
        if USE_ORB:
            kporb = np.float32([pt.pt for pt in kporb])
            return (kps, des)
        else:
            kps = np.float32([pt.pt for pt in kps])
            return (kps, feat)
            
        #return tuple of kp and features

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
        
        if USE_FLANNE:
            
            index_params = dict(algorithm = 1, trees = 5)
            search_params = dict(checks=100)
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            matches_raw = flann.knnMatch(feat_1,feat_2, 2)
            
        else:
            # Compute raw matches
            match = cv2.DescriptorMatcher_create("BruteForce")
            # use .knnMatch(feat_1, feat_2, 2) method from DescriptorMatcher
            matches_raw = match.knnMatch(feat_1, feat_2, 2)

        # Initialize matches array
        matches = []

        # Loop over each match in raw matches
        for m in matches_raw:
            # Use Lowe's ratio test to ensure distance is within certain ratio
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Compute homography (requiers at least 4 matches)
        if len(matches) >= MIN_MATCH_COUNT:

            # Construct two sets of points
            if USE_ORB:
                pts1 = np.float32([kp_1[i].pt for (_, i) in matches])
                pts2 = np.float32([kp_2[i].pt for (i, _) in matches])
            else:
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
            return False

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
    ### Can use bash commands or python lib functions
    if not os.path.exists(FRAME_SAVE_LOC):
        os.makedirs(FRAME_SAVE_LOC)
    if not os.path.exists(CURR_STITCHED_LOC):
        os.makedirs(CURR_STITCHED_LOC)

 
def img_stitching_demo(show_vis=0):
#def main():
    create_folders()

    START_TIME = datetime.datetime.now()

    stitcher = ImgStitcher(1)
    stitcher.start(SHOW_VIS_OUTPUT=show_vis)
    
    END_TIME = datetime.datetime.now()
    STITCH_TIME = END_TIME - START_TIME
    print('Stitching Process Finished\n')
    print('Frames Stitched: '+str(stitcher.FRAME_COUNTER-1)+'\n')
    print('Elapsed Time: '+str(STITCH_TIME)+'\n')

    cv2.destroyAllWindows()

#if __name__ == '__main__':
#    main()

    
