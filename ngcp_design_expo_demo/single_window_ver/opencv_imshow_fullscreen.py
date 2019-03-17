# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:44:27 2017

@author: sakurai
"""


import numpy as np
import cv2
from screeninfo import get_monitors

def show_demo(window_name, image, WAIT):
    screen_id = 2
    is_color = False

    # get the size of the screen
    #screen = get_monitors('osx')
    #width, height = screen.width, screen.height

    width, height = 1920, 1080
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)
    cv2.resizeWindow(window_name, width, height)
    #cv2.setWindowProperty(window_name, cv2.WINDOW_NORMAL,
                          #cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(WAIT)
    #cv2.destroyAllWindows()
