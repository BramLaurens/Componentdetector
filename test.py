# 
# Course: Beeldherkenning - VEBEHERK
# Author Fabian Meijneken & Bram Laurens
# 
# A script that runs different tests on all photos in /photos/test_photos

import glob
import cv2 as cv
import numpy as np

TESTPHOTO_PATH = "photos/test_photos"

test_result_dict = {
    "photo_number"          : [],
    "photo_component_name"  : []
}
photo_count = 0


# Test 1 - Contour detection
# No Quantified result yet
def contour_detection(photo):
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

    cv.imshow("test1", imshow_array_small)
    cv.waitKey(0)
    return 1


# Edge detection - try multiple methods for edge detection
# No Quantified result yet
def edge_detection(photo):
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

    cv.imshow("test1", combined_all_small)
    cv.waitKey(0)
    return 1


# Create the necessary lists in the test_result_dict, every test needs a list.
# In this list the quantified result of each test is saved
test_result_dict["contour_detection"] = []
test_result_dict["edge_detection"] = []


# Run for every photo in the test
for file_name in glob.glob(TESTPHOTO_PATH + "/*.png"):
    # Read image
    photo = cv.imread(file_name)

    ##-- Pre processing --##
    # Crop out the turntable edges
    h, w, c = photo.shape
    photo = photo[0:h, 160:w-200]       


    ##-- Testing --##
    # test_result_dict["contour_detection"].append( contour_detection(photo))
    test_result_dict["edge_detection"].append( edge_detection(photo))


    ##-- Other test data (for plotting) --##
    # Save the current photo count in the same list position as the test results
    test_result_dict["photo_number"].append(photo_count)

    # Save the current Object name in the same list position as the test results
    file_component_name = file_name.split("\\")[-1]                     # Get the last string after "\"
    file_component_name = file_component_name.split("_")[0]             # Remove the .png
    test_result_dict["photo_component_name"].append(file_component_name)

    photo_count += 1

print("Total photos tested: {}".format(photo_count))