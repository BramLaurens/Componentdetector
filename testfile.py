from datetime import datetime       # This library provides functions related to time.
import glob                         # This library provides functions used to gather all the filepaths found in "\photos"
import os                           # This library provides a simple way to get the name of a file from a path.
import cv2 as cv                    # OpenCV provides vision algorithms and functions
import numpy as np                  # Numpy provides multiple handy functions to be used on (multiple dimension) arrays
import matplotlib.pyplot as plt     # Matplotlib provides function to visualize the gathered information
import math
import colorsys

TESTPHOTO_PATH = "photos/blauw_test"               # Path where photos are found
# TESTPHOTO_PATH = "train_photos/DATASET_BLUE_21091417"

ENABLE_VERBOSE = True


# Edge detection - try multiple methods for edge detection
def edge_detection_test(photo: cv.typing.MatLike):
    test_img = np.copy(photo)

    # Deze gehele test wordt in GRAY gedaan
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)

    # Blur with (5,5) kernel
    test_img_blur = cv.GaussianBlur(test_img, (5,5), 0)

    edges_canny =       cv.Canny(test_img_blur, 50, 150)
    edges_sobel =       cv.Sobel(test_img_blur, cv.CV_8U, 1, 1, ksize=5)
    edges_laplacian =   cv.Laplacian(test_img_blur, cv.CV_8U, ksize=5)

    combined_edges = np.concatenate((edges_canny, edges_sobel, edges_laplacian), axis=1)


    thresh_value = 254
    ret, edges_canny_thresh      = cv.threshold(edges_canny, thresh_value, 255, cv.THRESH_BINARY)
    ret, edges_sobel_thresh      = cv.threshold(edges_sobel, thresh_value, 255, cv.THRESH_BINARY)
    ret, edges_laplacian_thresh  = cv.threshold(edges_laplacian, thresh_value, 255, cv.THRESH_BINARY)

    combined_edges_thresh = np.concatenate((edges_canny_thresh, edges_sobel_thresh, edges_laplacian_thresh), axis=1)

    combined_all = np.concatenate((combined_edges, combined_edges_thresh), axis=0)


    # Scale the result so that it can be viewed in whole
    original_height, original_width = combined_all.shape
    combined_all_small = cv.resize(combined_all, (int(original_width/2), int(original_height/2)))

    if ENABLE_VERBOSE:   
        cv.imshow("Edge detection test", combined_all_small)
        cv.waitKey(0)

    return 1

# Edge detection - try multiple parameters for a laplacian function
def laplacian_param_test(photo: cv.typing.MatLike):
    test_img = np.copy(photo)
    
    # Deze gehele test wordt in GRAY gedaan
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)

    # Blur with (5,5) kernel
    test_img_blur5 = cv.GaussianBlur(test_img, (5,5), 0)
    test_img_blur7 = cv.GaussianBlur(test_img, (7,7), 0)
    

    edges_laplacian_blur7 =   cv.Laplacian(test_img_blur7, cv.CV_8U, ksize=5)
    edges_laplacian_blur5 =   cv.Laplacian(test_img_blur5, cv.CV_8U, ksize=5)
    edges_laplacian_noblur =   cv.Laplacian(test_img, cv.CV_8U, ksize=5)

    
    ret, edges_laplacian_blur7_thresh = cv.threshold(edges_laplacian_blur7, 254, 255, cv.THRESH_BINARY)
    ret, edges_laplacian_blur5_thresh = cv.threshold(edges_laplacian_blur5, 254, 255, cv.THRESH_BINARY)
    ret, edges_laplacian_noblur_thresh  = cv.threshold(edges_laplacian_noblur, 254, 255, cv.THRESH_BINARY)


    combined_laplacian = np.concatenate((edges_laplacian_blur7, edges_laplacian_blur5, edges_laplacian_noblur), axis=1)
    combined_laplacian_thresh = np.concatenate((edges_laplacian_blur7_thresh, edges_laplacian_blur5_thresh, edges_laplacian_noblur_thresh), axis=1)
    combined_total = np.concatenate((combined_laplacian, combined_laplacian_thresh), axis=0)

    # Scale the result so that it can be viewed in whole
    original_height, original_width = combined_total.shape
    combined_all_small = cv.resize(combined_total, (int(original_width/2), int(original_height/2)))

    if ENABLE_VERBOSE:
        cv.imshow("Laplacian test", combined_all_small)
        cv.waitKey(0)
        
    return 1


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


    s = cv.convertScaleAbs(s, alpha=1.1, beta=0)                    # Increase contrast by 1.8
    ret, thresh = cv.threshold(s, 100, 255, cv.THRESH_BINARY)

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

# https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_histograms/py_histogram_begins/py_histogram_begins.html#find-histogram
def histogram_test(photo: cv.typing.MatLike):
    photo_copy = np.copy(photo)
    photo_gray = cv.cvtColor(photo_copy, cv.COLOR_BGR2GRAY)

    histogram = cv.calcHist([photo_gray], [0], None, [256], [0, 256])

    plt.plot(histogram)
    cv.imshow("Histogram_img", photo_gray)
    plt.draw()
    plt.waitforbuttonpress()
    # plt.close()





# === This is needed for find_pins_test() === ###
def nothing(a):
    return None

# cv.namedWindow("TestWindow", )
# cv.createTrackbar("B_low", "TestWindow", 100, 255, nothing)
# cv.createTrackbar("G_low", "TestWindow", 120, 255, nothing)
# cv.createTrackbar("R_low", "TestWindow", 180, 255, nothing)
# cv.createTrackbar("B_high", "TestWindow", 140, 255, nothing)
# cv.createTrackbar("G_high", "TestWindow", 255, 255, nothing)
# cv.createTrackbar("R_high", "TestWindow", 255, 255, nothing)
# cv.createTrackbar("Contrast", "TestWindow", 1000, 2000, nothing)

# B_low, G_low, R_low, Contrast = 100, 120, 180, 1
# B_high, G_high, R_high = 140, 255, 255
def find_pins_test(photo: cv.typing.MatLike):

    while(1):
        scope_photo = np.copy(photo)

        B_low = cv.getTrackbarPos("B_low", "TestWindow")
        G_low = cv.getTrackbarPos("G_low", "TestWindow")
        R_low = cv.getTrackbarPos("R_low", "TestWindow")
        B_high = cv.getTrackbarPos("B_high", "TestWindow")
        G_high = cv.getTrackbarPos("G_high", "TestWindow")
        R_high = cv.getTrackbarPos("R_high", "TestWindow")
        Contrast = cv.getTrackbarPos("Contrast", "TestWindow")

        
        scope_photo = cv.convertScaleAbs(scope_photo, alpha=(Contrast/1000), beta=0)                    # Increase contrast by 1.8

        # low_gray = np.array([0, 0, 200])
        # high_gray = np.array([255, 255, 255])

        low_gray = np.array([B_low, G_low, R_low])
        high_gray = np.array([B_high, G_high, R_high])
        

        mask = cv.inRange(scope_photo, low_gray, high_gray)


        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

        photo_mask = np.copy(scope_photo)
        ret = cv.bitwise_and(photo_mask, photo_mask, mask=mask_closed)


        cv.imshow("TestWindow", ret)
        # cv.imshow("Original image", scope_photo)
        # cv.imshow("Mask", mask_closed)
        # cv.imshow("Photo mask", ret)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
    

def IC_pincount(photo: cv.typing.MatLike):
    CONTOUR_AREA_THRESH = 50
    contour_count = 0

    # Create filter
    low_gray = np.array([0, 0, 200])
    high_gray = np.array([255, 255, 255])

    # Create mask from photo 
    mask = cv.inRange(photo, low_gray, high_gray)

    # Find contours in photo in order to count them
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Count all "Valid" contours. A "Valid" contour has a minimum area of CONTOUR_AREA_THRESH
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) >= CONTOUR_AREA_THRESH:
            contour_count += 1

    # Optional output
    if ENABLE_VERBOSE:
        print("Valid Contours found: ", contour_count)
        contours_img = np.copy(photo)
        contours_img = cv.drawContours(contours_img, contours, -1, (0, 0, 255), thickness=2)

        cv.imshow("Original image", photo)
        cv.imshow("Mask", mask)
        cv.imshow("Contours", contours_img)

        cv.waitKey(0)

    # Return value
    return contour_count
    

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

    # HSV_laplacian_edge_detection_test(photo)
    # colorspace_test(photo)
    # LAB_test(photo)
    # LAB_morphology_OPEN_test(photo)
    # LAB_corner_filter_test(photo)
    # histogram_test(photo)


    find_pins_test(photo)
    # IC_pincount(photo)
