# Filename:     autotest_script.py
# Author:       Fabian Meijneken & Bram Laurens
# University:   Utrecht University of Applied Sciences
# Course:       Beeldherkenning - VEBEHERK
# Project:      Electrical Component detector
#
# Description:  This script applies multiple vision function to all photos in a defined folder. 
#               The goal of this file is to extract certain parameters from the electrical components found in the image.
#               With this informaten a decision engine can be used to determine what electrical component is inside the image.

from datetime import datetime       # This library provides functions related to time.
import glob                         # This library provides functions used to gather all the filepaths found in "\photos"
import os                           # This library provides a simple way to get the name of a file from a path.
import cv2 as cv                    # OpenCV provides vision algorithms and functions
import numpy as np                  # Numpy provides multiple handy functions to be used on (multiple dimension) arrays
import matplotlib.pyplot as plt     # Matplotlib provides function to visualize the gathered information


ENABLE_VERBOSE = True


# Variables 
TESTPHOTO_PATH = "photos/test_photos"   # Path where photos are found
photo_count = 0                         # Stores the total count of processed images
test_result_dict = {                    # Stores all the information about the processed images. More entries are created in the main function 
    "photo_number"          : [],
    "photo_component_name"  : []
}


# A very simple function that returns a string of formatted time.
def current_time():
    return datetime.now().strftime("%H:%M:%S.%f")


##-- First tests / parameter test (not used in final product)--##
# Contour detection - Contour detection
# No Quantified result yet
def contour_detection(photo: cv.typing.MatLike):
    # Make a copy of the photo to ensure not editing the original (might make program slower?)
    test_img = np.copy(photo)

    # Convert the image to grayscale
    test_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)

    # Set a threshold and create a black-white image
    ret, test_img_thresh = cv.threshold(test_img, 120, 255, cv.THRESH_BINARY_INV)

    # Find contours
    contours, hierarchy = cv.findContours(test_img_thresh, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_NONE)

    # Apply contours
    output_frame = test_img_thresh.copy()
    output_frame = cv.cvtColor(test_img_thresh, cv.COLOR_GRAY2BGR) 
    cv.drawContours(output_frame, contours, -1, (0, 0, 255), 2, cv.FILLED)

    # Convert some GRAY images to BGR for stacking
    thresh_BGR = cv.cvtColor(test_img_thresh, cv.COLOR_GRAY2BGR)
    test_img_BGR = cv.cvtColor(test_img, cv.COLOR_GRAY2BGR)

    # Stack images to create a readable interface
    stack_1 = np.concatenate((photo, test_img_BGR), axis=1)
    stack_2 = np.concatenate((thresh_BGR, output_frame), axis=1)
    combined_output = np.concatenate((stack_1, stack_2), axis=0)

    # Scale the result so that it can be viewed in whole
    original_height, original_width, c = combined_output.shape
    imshow_array_small = cv.resize(combined_output, (int(original_width/2), int(original_height/2)))

    if ENABLE_VERBOSE:
        cv.imshow("test1", imshow_array_small)
        cv.waitKey(0)

    return 1

# Edge detection - try multiple methods for edge detection
# No Quantified result yet
def edge_detection(photo: cv.typing.MatLike):
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
        cv.imshow("test1", combined_all_small)
        cv.waitKey(0)

    return 1

# Hough lines
# No Quantified result yet
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
        cv.imshow("test1", combined_all_small)
        cv.waitKey(0)
        
    return 1


##-- Parameter functions --##

# This function calculates the circumference in pixels.
# Known limitations:
#   - This function includes the connection arms found on all electrical components TODO Fix this
#   - After the laplacian function, a couple white "blobs" can be seen in the background, these are also counted in the total circumference count.
def circumference(photo: cv.typing.MatLike):
    # This function doesn't utelize color seen in the image
    photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

    # Blur the photo with a kerkel size of 5x5.
        # The "0" defines sigmaX to 0, which means the standard deviation in the X direction is 0.
        # This results in an even blur
    photo_blur = cv.GaussianBlur(photo_gray, (5,5), 0)

    # Apply OpenCV's laplacian formula to the photo. This function applies a laplacian Kernel, which measures (and highlights) a rapid changes in pixel intensity. 
    photo_laplacian = cv.Laplacian(photo_blur, cv.CV_8U, ksize=5)

    # Apply a threshold to the laplacian of the photo to get only the 255.
        # The cv.THRESH_BINARY, makes sure it is a standard threshold operation (pixel value > thresh? then pixel=white, else pixel=black)
    ret, laplacian_thresh = cv.threshold(photo_laplacian, 254, 255, cv.THRESH_BINARY)


    ### TODO: REMOVE OR FIX
    #   I tried to remove the connectors from the components leaving only the body. For this i tried using the contours function, but this didn't work because of the broken (holes) shape.
    #   To fix this i tried dilating and eroding the shape with differently sized kernels.
    ''' 
    ## DILATE
    # Make image thesame size as laplacian_thresh, but with all 0's
    kernel = np.ones((5,5), np.uint8)
    dialated_thresh = cv.dilate(laplacian_thresh, kernel, iterations=2)

    ## ERODE
    kernel = np.ones((2,2), np.uint8)
    eroded_thresh = cv.erode(laplacian_thresh, kernel, iterations=1)

    # Contours
    contours, hierarchy = cv.findContours(laplacian_thresh, mode = cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    photo_contours = np.copy(laplacian_thresh)
    cv.drawContours(photo_contours, contours, -1, (0, 0, 255), 2, cv.FILLED)

    cv.imshow("circumference", laplacian_thresh)
    # cv.imshow("edge", photo_contours)
    # cv.imshow("dialated", dialated_thresh)
    cv.imshow("eroded", eroded_thresh)
    cv.waitKey(0)
    '''

    # TODO: REMOVE
    # Test to remove small blobs
    '''
    # contours, hierarchy = cv.findContours(laplacian_thresh, mode = cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    # photo_contours = np.copy(photo)
    # cv.drawContours(photo_contours, contours, -1, (0, 0, 255), 2, cv.FILLED)
    # cv.imshow("contour_test", photo_contours)
    '''



    # Calculate the circumference by counting all the non zero pixels in the image.
    #   This works since the above threshold function creates a strict black(255) white(0) picture.
    circumference = cv.countNonZero(laplacian_thresh)


    if ENABLE_VERBOSE:
        print("Circunference: {}".format(circumference))

        cv.imshow("circumference", laplacian_thresh)
        cv.waitKey(0)
   
    return circumference




print("{} - Program started".format(current_time()))

# Create the necessary lists in the test_result_dict, every test needs a list.
# In this list the quantified result of each test is saved
test_result_dict["contour_detection"] = []
test_result_dict["edge_detection"] = []
test_result_dict["laplacian_param_test"] = []
test_result_dict["circumference"] = []



# Run for every photo in the test
for file_path in glob.glob(TESTPHOTO_PATH + "/*.png"):
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

    ##-- Parameter testing --##
    # test_result_dict["contour_detection"].append( contour_detection(photo))
    # test_result_dict["edge_detection"].append( edge_detection(photo))
    # test_result_dict["laplacian_param_test"].append( laplacian_param_test(photo))

    ##-- Parameter functions --##
    test_result_dict["circumference"].append(circumference(photo))


    ##-- Other test data (for plotting) --##
    # Save the current photo count in the same list position as the test results
    test_result_dict["photo_number"].append(photo_count)

    # Save the current Object name in the same list position as the test results
    file_component_name = os.path.basename(file_path)
    file_component_name = file_component_name.split("_")[0]
    test_result_dict["photo_component_name"].append(file_component_name)

    photo_count += 1


print("{} - Done calculating. Total photos tested: {}".format(current_time(), photo_count))
if ENABLE_VERBOSE: cv.destroyAllWindows()           # Sometimes the last window gets left behind, destroy all windows

print("{} - Starting result visualisation".format(current_time()))


##-- Results --##
##-- Visualisation --#
# Use a subplot for futureproofing, alowing for more plots in 1 window in the future.
#   If this is done "axs." needs to be replaced with "axs[x]." with x representing the plot
fig, axs = plt.subplots(1)

# Plot the component name vs circumference
axs.scatter(test_result_dict["photo_component_name"], test_result_dict["circumference"])
axs.set_title("Circumference per component")
axs.set(xlabel="Component", ylabel="Circumference (px)")

# Show the figures
plt.show()


print("{} - Program Done".format(current_time()))