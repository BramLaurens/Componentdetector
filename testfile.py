from datetime import datetime       # This library provides functions related to time.
import glob                         # This library provides functions used to gather all the filepaths found in "\photos"
import os                           # This library provides a simple way to get the name of a file from a path.
import cv2 as cv                    # OpenCV provides vision algorithms and functions
import numpy as np                  # Numpy provides multiple handy functions to be used on (multiple dimension) arrays
import matplotlib.pyplot as plt     # Matplotlib provides function to visualize the gathered information
import math
import colorsys

# TESTPHOTO_PATH = "photos/blauw_test"               # Path where photos are found
TESTPHOTO_PATH = "photos/train_photos/DATASET_BLUE_singles"

ENABLE_VERBOSE = False


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
    # HSV
    photo_HSV = cv.cvtColor(photo, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(photo_HSV)
    combined_hsv = np.concatenate((h, s, v), axis=1)
    c_hsv_height, c_hsv_width = combined_hsv.shape
    combined_hsv_small = cv.resize(combined_hsv, (int(c_hsv_width/2), int(c_hsv_height/2)))
    cv.imshow("HSV - split", combined_hsv_small)
    cv.imshow("HSV", photo_HSV)

    # HLS
    photo_HLS = cv.cvtColor(photo, cv.COLOR_BGR2HLS)
    h, l, s = cv.split(photo_HSV)
    combined_hls = np.concatenate((h, l, s), axis=1)
    c_hls_height, c_hls_width = combined_hsv.shape
    combined_hls_small = cv.resize(combined_hls, (int(c_hls_width/2), int(c_hls_height/2)))
    cv.imshow("HLS - split", combined_hls_small)
    cv.imshow("HLS", photo_HLS) 

    # LAB
    photo_LAB = cv.cvtColor(photo, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(photo_LAB)
    combined_lab = np.concatenate((l, a, b), axis=1)
    c_lab_height, c_lab_width = combined_lab.shape
    combined_lab_small = cv.resize(combined_lab, (int(c_lab_width/2), int(c_lab_height/2)))
    cv.imshow("LAB - split", combined_lab_small)
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


    s = cv.convertScaleAbs(s, alpha=1.1, beta=0)                    # Increase contrast by 1.1 to make some connectors more visible
    ret, thresh = cv.threshold(s, 100, 255, cv.THRESH_BINARY)

    cv.imshow("Photo", photo)
    cv.imshow("HSV", photo_hsv)
    cv.imshow("HSV - H", h)
    cv.imshow("HSV - S", s)
    cv.imshow("HSV - V", v)
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
    # photo_gray = cv.cvtColor(photo_copy, cv.COLOR_BGR2GRAY)

    # histogram = cv.calcHist([photo_copy], [0], None, [256], [0, 256])
    # histogram = cv.calcHist([photo_copy], [0], None, [256], [0, 256])

    # plt.plot(histogram, color="blue")
    # cv.imshow("Histogram_img", photo_copy)
    # plt.draw()
    # plt.waitforbuttonpress()
    # plt.close()
    colors = ('b','g','r')
    for i,color in enumerate(colors):
        hist = cv.calcHist([photo_copy],[i],None,[256],[0,256])
        plt.plot(hist,color = color)
    plt.title('Image Histogram GFG')
    plt.draw()
    plt.waitforbuttonpress()


def count_unique_colors(photo: cv.typing.MatLike):
    # unique_colors = np.unique(photo)

    photo_copy = np.copy(photo)
    # Reshape the 3D array to a 2D array merging the first two dimensions
    Ar = photo_copy.reshape(-1,photo_copy.shape[2])

    # Perform lex sort and get the sorted indices and xy pairs
    sorted_idx = np.lexsort(Ar.T)
    sorted_Ar =  Ar[sorted_idx,:]

    # Get the count of rows that have at least one TRUE value 
    # indicating presence of unique subarray there
    unq_out = np.any(np.diff(sorted_Ar,axis=0),1).sum()+1

    print(unq_out)

# === This is needed for find_pins_test() === ###
def nothing(a):
    return None

# cv.namedWindow("TestWindow")
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


        print(IC_pincount(ret))

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

        # cv.waitKey(0)

    # Return value
    return contour_count

    
def circumference(photo: cv.typing.MatLike):
    # This function doesn't utelize color seen in the image
    photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

    # Blur the photo with a kerkel size of 5x5.
        # The "0" defines sigmaX to 0, which means the standard deviation in the X direction is 0.
        # This results in an even blur
    photo_blur = cv.GaussianBlur(photo_gray, (3,3), 0)

    # Apply OpenCV's laplacian formula to the photo. This function applies a laplacian Kernel, which measures (and highlights) a rapid changes in pixel intensity. 
    photo_laplacian = cv.Laplacian(photo_blur, cv.CV_8U, ksize=5)
    
    # Apply a threshold to the laplacian of the photo to get only the 255.
        # The cv.THRESH_BINARY, makes sure it is a standard threshold operation (pixel value > thresh? then pixel=white, else pixel=black)
    ret, laplacian_thresh = cv.threshold(photo_laplacian, 254, 255, cv.THRESH_BINARY)

    # Calculate the circumference by counting all the non zero pixels in the image.
    #   This works since the above threshold function creates a strict black(255) white(0) picture.
    circumference = cv.countNonZero(laplacian_thresh)


    if ENABLE_VERBOSE:
        cv.imshow("laplacian", laplacian_thresh)
        print("Circumference: {}".format(circumference))
        cv.waitKey(0)
   
    return circumference


# This function creates a snippit out of the original photo which only includes the component
# Known limitation: Function currently only works for photos with a single component. If multiple components are visible, it will crop around both components.
def crop_to_component(photo: cv.typing.MatLike):
    PADDING = 20        # Padding is applied around the object to ensure the entire object is captured.

    # Convert photo to HSV and apply a blur prevent white spots in threshold
    photo_HSV = cv.cvtColor(photo, cv.COLOR_BGR2HSV)
    photo_HSV_blur = cv.GaussianBlur(photo_HSV, (15,15), 0)

    # Split the H S V values and apply a threshold to the s channel. This threshold value (110), was found by trial and error
    h, s, v = cv.split(photo_HSV_blur)
    ret, s_thresh = cv.threshold(s, 110, 255, cv.THRESH_BINARY)

    # Calculate the bounding rectangle (smallest rectangle that contains all white pixels)
    x, y, w, h = cv.boundingRect(s_thresh)

    # Add padding to bounding rectangle
    y_start_padding = max(0, y - PADDING)
    y_end_padding = min(photo.shape[0], y + h + PADDING)
    x_start_padding = max(0, x - PADDING)
    x_end_padding = min(photo.shape[1], x + w + PADDING)

    # Crop the original photo with the calculated bounding rectangle and padding
    photo_cropped = photo[y_start_padding:y_end_padding, x_start_padding:x_end_padding]

    if ENABLE_VERBOSE:
        # Draw bounding rectangle with padding
        photo_HSV_rect = np.copy(photo)
        cv.rectangle(photo_HSV_rect, (x_start_padding, y_start_padding), (x_end_padding, y_end_padding), (0, 255, 0))

        cv.imshow("HSV - S with threshold", s_thresh)
        cv.imshow("Bounding rectangle", photo_HSV_rect)
        cv.imshow("Cropped result", photo_cropped)

        cv.waitKey(0)

    # Return the cropped photo
    return photo_cropped


def body_mask(photo: cv.typing.MatLike):
    # Convert photo to HSV and apply a blur prevent white spots in threshold
    photo_HSV = cv.cvtColor(photo, cv.COLOR_BGR2HSV)
    photo_HSV_blur = cv.GaussianBlur(photo_HSV, (15,15), 0)

    h, s, v = cv.split(photo_HSV_blur)

    s = cv.convertScaleAbs(s, alpha=1.1, beta=0)                    # Increase contrast by 1.1 to make some connectors more visible
    ret, thresh = cv.threshold(s, 100, 255, cv.THRESH_BINARY)

    body_masked = cv.bitwise_and(photo, photo, mask=thresh)

    cv.imshow("mask", body_masked)
    cv.waitKey(0)

    return body_masked


# def frequency_domain_test(photo: cv.typing.MatLike):
#     photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

#     # 1. Grayscale FFT
#     photo_fft = np.fft.fft2(photo_gray)
#     photo_fft_shift = np.fft.fftshift(photo_fft)

#     # Calculate magnitude spectrum for grayscale (log scale to make frequencies visible)
#     magnitude_gray = 20 * np.log(np.abs(photo_fft_shift) + 1)
#     # Normalize to [0, 255] for OpenCV rendering
#     magnitude_gray = cv.normalize(magnitude_gray, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

#     ret, magnitude_thresh = cv.threshold(magnitude_gray, 150, 255, cv.THRESH_BINARY)

#     fft_BGR = cv.cvtColor(magnitude_thresh, cv.COLOR_GRAY2BGR)



#     lines = cv.HoughLinesP(
#         magnitude_thresh, 
#         rho=1, 
#         theta=np.pi / 180, 
#         threshold=40, 
#         minLineLength=50, 
#         maxLineGap=20
#     )


#     if ENABLE_VERBOSE:
#         if lines is not None:
#             for line in lines:
#                 # Flatten array to handle shape variations safely
#                 line_data = line.ravel()
                
#                 if len(line_data) == 4:
#                     # Probabilistic Hough result: (x1, y1, x2, y2)
#                     x1, y1, x2, y2 = line_data
#                     cv.line(fft_BGR, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
#                 elif len(line_data) == 2:
#                     # Standard Hough fallback: (rho, theta)
#                     rho, theta = line_data
#                     a, b = np.cos(theta), np.sin(theta)
#                     x0, y0 = a * rho, b * rho
#                     x1 = int(x0 + 1000 * (-b))
#                     y1 = int(y0 + 1000 * (a))
#                     x2 = int(x0 - 1000 * (-b))
#                     y2 = int(y0 - 1000 * (a))
#                     cv.line(fft_BGR, (x1, y1), (x2, y2), (0, 0, 255), 2)


#         cv.imshow('Grayscale FFT Magnitude', fft_BGR)
#         cv.imshow('Grayscale Image', photo_gray)
    
#         cv.waitKey(0)

#     if lines is not None:
#         return len(lines)
#     else:
#         return 0
#     # return len(lines) if lines is not None else 0
    


def frequency_domain_test(photo: cv.typing.MatLike):
    photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

    # 1. Grayscale FFT
    photo_fft = np.fft.fft2(photo_gray)
    photo_fft_shift = np.fft.fftshift(photo_fft)

    # Save original phase angle (REQUIRED for inverse transform)
    phase = np.angle(photo_fft_shift)

    # Calculate magnitude spectrum for grayscale (log scale to make frequencies visible)
    magnitude_gray = 20 * np.log(np.abs(photo_fft_shift) + 1)

    log_min = magnitude_gray.min()
    log_max = magnitude_gray.max()   
    magnitude_gray = cv.normalize(magnitude_gray, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    ret, magnitude_thresh = cv.threshold(magnitude_gray, 150, 255, cv.THRESH_BINARY)

    # Black out low frequencies (High-Pass Filter)
    # Note: make a copy if you want to keep the unedited magnitude_gray separate
    circle_size = min(magnitude_gray.shape[0], magnitude_gray.shape[1]) // 3
    cv.circle(magnitude_gray, (magnitude_gray.shape[1]//2, magnitude_gray.shape[0]//2), circle_size, 0, thickness=-1)

    # --- REVERSE TRANSFORM ---
    # Step 1: Scale back from [0, 255] to log magnitude range
    restored_log = (magnitude_gray.astype(np.float64) / 255.0) * (log_max - log_min) + log_min

    # Step 2: Inverse of 20 * log(x + 1)
    restored_magnitude = np.exp(restored_log / 20.0) - 1.0

    # Step 3: Combine modified magnitude with original phase using Euler's formula: mag * exp(1j * phase)
    modified_fft_shift = restored_magnitude * np.exp(1j * phase)

    # Step 4: Shift back and perform 2D Inverse FFT
    modified_fft = np.fft.ifftshift(modified_fft_shift)
    inverse_complex = np.fft.ifft2(modified_fft)

    # Step 5: Convert complex numbers to magnitude & normalize for OpenCV display
    inverse_img = np.abs(inverse_complex)
    inverse_img = cv.normalize(inverse_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Display results
    cv.imshow('Grayscale FFT Magnitude', magnitude_thresh)
    cv.imshow('FFT Circle Mask Applied', magnitude_gray)
    cv.imshow('Inverse Reconstruction', inverse_img)
    cv.imshow('photo', photo_gray)
    cv.waitKey(0)
    # fft_BGR = cv.cvtColor(magnitude_thresh, cv.COLOR_GRAY2BGR)



def hsv_derivative_test(photo: cv.typing.MatLike):
    hsv_photo = cv.cvtColor(photo, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv_photo)

    h_hist = cv.calcHist([h], [0], None, [256], [0, 256])
    

    # moving average
    MAF_k_size = 20
    MAF_kernel = np.ones(MAF_k_size) / MAF_k_size       # Create kernel that sums up to 1
    h_hist_MAF = np.copy(h_hist)

    h_hist_MAF = np.convolve(h_hist, MAF_kernel, mode='same')

    h_hist_MAF_thresh = np.copy(h_hist_MAF)

    # Calculate total
    h_hist_MAF_total = np.sum(h_hist_MAF)
    print("MAF TOTAL: {}".format(h_hist_MAF_total))
    print("total pixels: {}".format(photo.shape[0] * photo.shape[1]))

    for i in range(len(h_hist_MAF)):
        if h_hist_MAF[i] >= 500:
            h_hist_MAF_thresh[i] = 1
        else:
            h_hist_MAF_thresh[i] = 0

    
    # deriv
    h_hist_deriv = np.gradient(h_hist_MAF)

    
    # visualisation
    fig_h, ax_h = plt.subplots()
    ax2_h = ax_h.twinx()
    ax3_h = ax_h.twinx()


    ax_h.cla()
    ax2_h.cla()
    ax3_h.cla()
    # fig.canvas.manager.set_window_title('HSV Derivative Test')
    
    ax_h.plot(h_hist, color='r', alpha=0.5)
    ax2_h.plot(h_hist_MAF, color='r', alpha=1, linestyle='--')
    ax3_h.plot(h_hist_MAF_thresh, color='b', alpha=1)
    
    cv.imshow("photo", photo)
    plt.show(block=False)
    
    cv.waitKey(0)

    plt.close(fig_h)



    
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

    photo_cropped = crop_to_component(photo)
    # body_mask(photo_cropped)


    file_component_name = os.path.basename(file_path)
    file_component_name = file_component_name.split("_")[0]

    # frequency_domain_test(photo_cropped)
    hsv_derivative_test(photo_cropped)

    # HSV_laplacian_edge_detection_test(photo)
    # colorspace_test(photo)
    # LAB_test(photo)
    # LAB_morphology_OPEN_test(photo)
    # LAB_corner_filter_test(photo)
    # histogram_test(photo)


    # find_pins_test(photo)
    # IC_pincount(photo)
    # histogram_test(photo)

    # count_unique_colors(photo)
    # circumference(photo)
    # circumference2(photo)

