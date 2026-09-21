from datetime import datetime       # This library provides functions related to time.
import glob                         # This library provides functions used to gather all the filepaths found in "\photos"
import os                           # This library provides a simple way to get the name of a file from a path.
import cv2 as cv                    # OpenCV provides vision algorithms and functions
import numpy as np                  # Numpy provides multiple handy functions to be used on (multiple dimension) arrays
import matplotlib.pyplot as plt     # Matplotlib provides function to visualize the gathered information
import math
import colorsys

TESTPHOTO_PATH = "photos/Trainset_150901"               # Path where photos are found



def colorspace_test(photo: cv.typing.MatLike):
    photo_HSV = cv.cvtColor(photo, cv.COLOR_BGR2HSV)
    cv.imshow("HSV", photo_HSV)

    photo_HLS = cv.cvtColor(photo, cv.COLOR_BGR2HLS)
    cv.imshow("H:S", photo_HLS)

    photo_LAB = cv.cvtColor(photo, cv.COLOR_BGR2LAB)
    cv.imshow("LAB", photo_LAB)

    cv.waitKey(0)


def HSV_laplacian_edge_detection_test(photo: cv.typing.MatLike):
    photo_hsv = cv.cvtColor(photo, cv.COLOR_BGR2HSV)
    photo_BGR2 = cv.cvtColor(photo_hsv, cv.COLOR_HSV2BGR)

    # Blur the photo with a kerkel size of 5x5.
        # The "0" defines sigmaX to 0, which means the standard deviation in the X direction is 0.
        # This results in an even blur
    photo_blur = cv.GaussianBlur(photo_hsv, (5,5), 0)

    # Apply OpenCV's laplacian formula to the photo. This function applies a laplacian Kernel, which measures (and highlights) a rapid changes in pixel intensity. 
    photo_laplacian = cv.Laplacian(photo_blur, cv.CV_32F, ksize=5)

    # Apply a threshold to the laplacian of the photo to get only the 255.
        # The cv.THRESH_laplacian_thresh, makes sure it is a standard threshold operation (pixel value > thresh? then pixel=white, else pixel=black)
    ret, laplacian_thresh = cv.threshold(photo_laplacian, 254, 255, cv.THRESH_BINARY)



    # Define filters in RGB
    lower_filter = np.array([0, 195, 195])
    higer_filter = np.array([255,255,255])


    mask = cv.inRange(photo_hsv, lowerb=lower_filter, upperb=higer_filter)

    # kernel_opening = np.ones((2,2), np.uint8)
    # mask_test = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=kernel_opening)

    # mask_image = np.copy(laplacian_thresh)
    # cv.bitwise_and(mask_image, mask_image, mask=mask)
    
    # kernel_opening = np.zeros((9,9), np.uint8)
    # mask_opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel=kernel_opening)

    # Show result
    # cv.imshow("Photo", photo)
    # cv.imshow("laplacian", laplacian_thresh)
    # cv.imshow("Mask", mask)

    cv.imshow("HSV", photo_hsv)
    cv.imshow("BGR", photo_BGR2)

    # cv.imshow("Mask_test", mask_test)
    cv.waitKey(0)


def HSV_test(photo: cv.typing.MatLike):
    photo_hsv = cv.cvtColor(photo, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(photo_hsv)
    # Define filters in RGB
    # lower_filter = np.array([0, 0, 150])
    # higer_filter = np.array([255,255,255])

    # mask = cv.inRange(photo_hsv, lowerb=lower_filter, upperb=higer_filter)
    # cv.imshow("HSV", photo_hsv)
    # cv.imshow("H", h)
    # cv.imshow("S", s)
    # cv.imshow("V", v)
    # cv.imshow("BGR", photo)
    # cv.imshow("hsv", photo_hsv)


    s = cv.convertScaleAbs(s, alpha=1.8, beta=0)                    # Increase contrast by 1.8
    ret, thresh = cv.threshold(s, 200, 255, cv.THRESH_BINARY)

    cv.imshow("Photo", photo)
    cv.imshow("HSV", photo_hsv)
    cv.imshow("HSV - S", s)
    cv.imshow("Mask", thresh)


    canny = cv.Canny(thresh, 50, 150)
    cv.imshow("Canny", canny)

    cv.waitKey(0)


def LAB_test(photo: cv.typing.MatLike):
    photo_lab = cv.cvtColor(photo, cv.COLOR_BGR2LAB)

    l, a, b = cv.split(photo_lab)

    l_contrast = cv.convertScaleAbs(l, alpha=2, beta=0)                    # Increase contrast by 1.8
    ret, thresh = cv.threshold(l, 110, 255, cv.THRESH_BINARY)

    cv.imshow("L", l)
    cv.imshow("A", a)
    cv.imshow("B", b)
    # cv.imshow("L Contrast", l_contrast)

    cv.imshow("LAB", photo_lab)
    cv.imshow("Photo", photo)
    # cv.imshow("L", l)
    cv.imshow("Mask", thresh)


    # canny = cv.Canny(s, 50, 150)
    # cv.imshow("Canny", canny)

    cv.waitKey(0)


# derrived from LAB_test() 
def LAB_morphology_OPEN_test(photo: cv.typing.MatLike):
    photo_lab = cv.cvtColor(photo, cv.COLOR_BGR2LAB)
    
    l, a, b = cv.split(photo_lab)

    l_contrast = cv.convertScaleAbs(l, alpha=2, beta=0)                    # Increase contrast by 1.8
    ret, thresh = cv.threshold(l, 100, 255, cv.THRESH_BINARY_INV)

    kernel_square = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
    kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (19, 19))

    thresh_close_ellipse = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel_ellipse)

    # thresh_open_square = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_square)
    # thresh_open_ellipse = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_ellipse)

    kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2))

    output_contour_frame = np.copy(photo)
    contours, hierarchy = cv.findContours(thresh_close_ellipse, mode = cv.RETR_EXTERNAL, method = cv.CHAIN_APPROX_NONE)
    cv.drawContours(output_contour_frame, contours, -1, (0, 0, 255), 2, cv.FILLED)




    # https://stackoverflow.com/questions/11782147/python-opencv-contour-tree-hierarchy-structure
    # if contours is not None:
    #     hierarchy = hierarchy[0]

    #     for component in zip(contours, hierarchy):
    #         currentContour = component[0]
    #         currentHierarchy = component[1]
    #         x,y,w,h = cv.boundingRect(currentContour)
    #         if currentHierarchy[2] < 0:
    #             # these are the innermost child components
    #             cv.rectangle(output_contour_frame,(x,y),(x+w,y+h),(0,0,255),1)
    #         elif currentHierarchy[3] < 0:
    #             # these are the outermost parent components
    #             cv.rectangle(output_contour_frame,(x,y),(x+w,y+h),(0,255,0),1)



    cv.imshow("thresh", thresh)
    cv.imshow("close Ellipse", thresh_close_ellipse)
    cv.imshow("contour", output_contour_frame)

    # cv.imshow("thresh_open_square", thresh_open_square)
    # cv.imshow("thresh_open_ellipse", thresh_open_ellipse)
    # cv.imshow("Gradient", output_contour_frame)

    cv.waitKey(0)


# Derrived from LAB_morphology_OPEN_test()
def LAB_corner_filter_test (photo: cv.typing.MatLike):
    photo_lab = cv.cvtColor(photo, cv.COLOR_BGR2LAB)
    
    l, a, b = cv.split(photo_lab)

    l_contrast = cv.convertScaleAbs(l, alpha=2, beta=0)                    # Increase contrast by 1.8
    ret, thresh = cv.threshold(l, 100, 255, cv.THRESH_BINARY_INV)

    kernel_square = cv.getStructuringElement(cv.MORPH_RECT,(7,7))
    kernel_ellipse = cv.getStructuringElement(cv.MORPH_ELLIPSE, (19, 19))

    thresh_close_ellipse = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel_ellipse)

    # thresh_open_square = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_square)
    # thresh_open_ellipse = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel_ellipse)

    cv.imshow("thresh", thresh)
    cv.imshow("close Ellipse", thresh_close_ellipse)


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


    file_component_name = os.path.basename(file_path)
    file_component_name = file_component_name.split("_")[0]

    if file_component_name == "ElecCapacitor_20260911-105907.jpg":
        print("STOP!")

    # HSV_laplacian_edge_detection_test(photo)
    # colorspace_test(photo)
    # HSV_test(photo)
    # LAB_test(photo)
    # LAB_morphology_OPEN_test(photo)
    LAB_corner_filter_test(photo)