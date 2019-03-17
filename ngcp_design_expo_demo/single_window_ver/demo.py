import cv2
import imutils
from cv_img_stitching import img_stitching_demo
from cv_img_read import img_poi_detection_demo
from time import sleep
from opencv_imshow_fullscreen import show_demo



def main():

	mode = input('Select Demo Mode: With Visual [O]utput, [T]ime test mode\n')

	boolean = 0

	if (mode == 'O'):
		boolean = 1

	if (mode == 'T'):
		boolean = 0

	while True:
		if (boolean):
			
			intro_s = cv2.imread('./image_stitching_intro.jpg')
			show_demo('IMAGE STITCHING', intro_s, 1050)


			#cv2.imshow('IMAGE STITCHING', intro_s)

			#cv2.namedWindow('IMAGE STITCHING', cv2.WINDOW_NORMAL)
			#cv2.resizeWindow('IMAGE STITCHING', 1920, 1080)
			#cv2.moveWindow('IMAGE STITCHING', 0,0)
			
			#cv2.waitKey(1050)
			#cv2.destroyWindow('IMAGE STITCHING')

		img_stitching_demo(boolean)
		cv2.destroyAllWindows()
		sleep(2)

		if (boolean):
			
			intro_p = cv2.imread('./poi_analysis_intro.jpg')

			show_demo('POI ANALYSIS', intro_p, 1050)


			#cv2.imshow('POI ANALYSIS', intro_p)

			#cv2.namedWindow('POI ANALYSIS', cv2.WINDOW_NORMAL)
			#cv2.resizeWindow('POI ANALYSIS', 1920, 1080)
			#cv2.moveWindow('POI ANALYSIS', 0,0)

			#cv2.waitKey(1050)
			#cv2.destroyWindow('POI ANALYSIS')

		img_poi_detection_demo(boolean)
		cv2.destroyAllWindows()
		sleep(2)

		if boolean == 0:
			exit(0)
			break;



if __name__ == '__main__':
	main()