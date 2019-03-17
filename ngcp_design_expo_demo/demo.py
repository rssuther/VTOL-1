import cv2
import imutils
from cv_img_stitching import img_stitching_demo
from cv_img_read import img_poi_detection_demo
from time import sleep
import sys




def main():

	if len(sys.argv) > 1:
		mode_v = sys.argv[1]
		mode_d = sys.argv[2]

		if mode_d == 'S':
			mode_p = sys.argv[3]

	else:

		mode_v = input('Select Demo Mode: With Visual [O]utput, [T]ime test mode\n')
		mode_d = input('Select Demo Mode: [D]ual output, [S]ingle mode\n')
		if mode_d == 'S':
			mode_p = input('Select Process for Dual Mode: [S]titching, [P]OI\n')

	boolean = 0

	if (mode_v == 'O'):
		boolean = 1

	if (mode_v == 'T'):
		boolean = 0


	if mode_d == 'S':

		if mode_p == 'S':

			while True:
				if (boolean):
					cv2.namedWindow('IMAGE STITCHING', cv2.WINDOW_AUTOSIZE)
					#cv2.resizeWindow('KEYPOINT MATCHES', 1920, 1080)
					cv2.moveWindow('IMAGE STITCHING', 150, 172)
					intro_s = cv2.imread('./image_stitching_intro.jpg')
					cv2.imshow('IMAGE STITCHING', intro_s)
					cv2.waitKey(1050)
					cv2.destroyWindow('IMAGE STITCHING')

				img_stitching_demo(boolean)
				sleep(2)

				if boolean == 0:
					break

		if mode_p == 'P':

			while True:
				if (boolean):
					cv2.namedWindow('POI ANALYSIS', cv2.WINDOW_AUTOSIZE)
					#cv2.resizeWindow('KEYPOINT MATCHES', 1920, 1080)
					cv2.moveWindow('POI ANALYSIS', 1100, 90)
					intro_p = cv2.imread('./poi_analysis_intro_s.jpg')
					cv2.imshow('POI ANALYSIS', intro_p)
					cv2.waitKey(1050)
					cv2.destroyWindow('POI ANALYSIS')

				img_poi_detection_demo(boolean)
				sleep(2)

				if boolean == 0:
					break

	if mode_d == 'D':

		while True:
			if (boolean):
				cv2.namedWindow('IMAGE STITCHING', cv2.WINDOW_AUTOSIZE)
				#cv2.resizeWindow('KEYPOINT MATCHES', 1920, 1080)
				cv2.moveWindow('IMAGE STITCHING', 150, 172)
				intro_s = cv2.imread('./image_stitching_intro.jpg')
				cv2.imshow('IMAGE STITCHING', intro_s)
				cv2.waitKey(1050)
				cv2.destroyWindow('IMAGE STITCHING')

			img_stitching_demo(boolean)
			sleep(2)

			if (boolean):
				cv2.namedWindow('POI ANALYSIS', cv2.WINDOW_AUTOSIZE)
			#cv2.resizeWindow('KEYPOINT MATCHES', 1920, 1080)
				cv2.moveWindow('POI ANALYSIS', 1100, 90)
				intro_p = cv2.imread('./poi_analysis_intro_s.jpg')
				cv2.imshow('POI ANALYSIS', intro_p)
				cv2.waitKey(1050)
				cv2.destroyWindow('POI ANALYSIS')

			img_poi_detection_demo(boolean)
			sleep(2)

			if boolean == 0:
				break;



if __name__ == '__main__':
	main()