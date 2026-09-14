from datetime import datetime       # This library provides functions related to time.
import glob                         # This library provides functions used to gather all the filepaths found in "\photos"
import os                           # This library provides a simple way to get the name of a file from a path.
import cv2 as cv                    # OpenCV provides vision algorithms and functions
import numpy as np                  # Numpy provides multiple handy functions to be used on (multiple dimension) arrays
import matplotlib.pyplot as plt     # Matplotlib provides function to visualize the gathered information
import math

TESTPHOTO_PATH = "photos"               # Path where photos are found



def colorspace_test(photo: cv.typing.MatLike):
    photo_HSV = cv.cvtColor(photo, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", photo_HSV)

    photo_HLS = cv.cvtColor(photo, cv.COLOR_BGR2HLS)
    cv.imshow("H:S", photo_HLS)

    photo_LAB = cv.cvtColor(photo, cv.COLOR_BGR2LAB)
    cv.imshow("LAB", photo_LAB)

    cv.waitKey(0)




def HSV_laplacian_edge_detection_test(photo: cv.typing.MatLike):
    photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2HSV)

    # Blur the photo with a kerkel size of 5x5.
        # The "0" defines sigmaX to 0, which means the standard deviation in the X direction is 0.
        # This results in an even blur
    photo_blur = cv.GaussianBlur(photo_gray, (5,5), 0)

    # Apply OpenCV's laplacian formula to the photo. This function applies a laplacian Kernel, which measures (and highlights) a rapid changes in pixel intensity. 
    photo_laplacian = cv.Laplacian(photo_blur, cv.CV_32F, ksize=5)

    # Apply a threshold to the laplacian of the photo to get only the 255.
        # The cv.THRESH_laplacian_thresh, makes sure it is a standard threshold operation (pixel value > thresh? then pixel=white, else pixel=black)
    ret, laplacian_thresh = cv.threshold(photo_laplacian, 254, 255, cv.THRESH_BINARY)

    mask = cv.inRange(laplacian_thresh, (0, 0, 0), (255, 255, 70))

    kernel_opening = np.zeros((100,100), np.uint8)

    mask_opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=kernel_opening)

    # Show result
    cv.imshow("Photo", photo)
    cv.imshow("Mask", mask)
    cv.imshow("Mask_opening", mask_opening)
    cv.waitKey(0)






# Quick lines of code to loop through all photos in "\photos"
for file_path in glob.glob(TESTPHOTO_PATH + "/*.jpg"):
    # Read image
    photo = cv.imread(file_path)

    # Check if the photo exists at given path. If not, go the the next item.
    # This check could be removed, but results in an "reportOptionalMemberAccess" pylance flag at "photo.shape", since we cannot prove it is never None.
    if photo is None:
        print("No image found at path: {}".format(file_path))
        continue
    
    ##-- Pre processing --##
    # Crop out the turntable edges
    h, w, c = photo.shape
    photo = photo[0:h, 160:w-200]   


    HSV_laplacian_edge_detection_test(photo)