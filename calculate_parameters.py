# Filename:     calculate_parameters.py
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

ENABLE_VERBOSE = False


# Variables 
TESTPHOTO_PATH = "photos/train_photos/DATASET_BLUE_21091417"               # Path where photos are found
photo_count = 0                         # Stores the total count of processed images
test_result_dict = {                    # Stores all the information about the processed images. More entries are created in the main function 
    "photo_number"          : [],
    "photo_component_name"  : []
}


# A function that returns a string of formatted time.
def current_time():
    return datetime.now().strftime("%H:%M:%S.%f")

# A function that returns the component name from the image path
def get_component_name(file_path: str):
    file_component_name = os.path.basename(file_path)
    return file_component_name.split("_")[0]


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
    photo_blur = cv.GaussianBlur(photo_gray, (3,3), 0)

    # Apply OpenCV's laplacian formula to the photo. This function applies a laplacian Kernel, which measures (and highlights) a rapid changes in pixel intensity. 
    photo_laplacian = cv.Laplacian(photo_blur, cv.CV_8U, ksize=5)

    # Apply a threshold to the laplacian of the photo to get only the 255.
        # The cv.THRESH_BINARY, makes sure it is a standard threshold operation (pixel value > thresh? then pixel=white, else pixel=black)
    ret, laplacian_thresh = cv.threshold(photo_laplacian, 254, 255, cv.THRESH_BINARY)



    ## TESTING ##

    '''
    lines = cv.HoughLinesP(
                laplacian_thresh, 
                rho=1, 
                theta=np.pi/180, 
                threshold=40, 
                minLineLength=150,  # <-- Crucial parameter!
                maxLineGap=60
            )

    output_hough = cv.cvtColor(laplacian_thresh, cv.COLOR_GRAY2BGR)

    if lines is not None:
        print(f"Found {len(lines)} line segments.")
        for line in lines:
            x1, y1, x2, y2 = line
            cv.line(output_hough, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)
        
    kernel_hough_open = np.ones((2,2),np.uint8)
    hough_open_img = cv.morphologyEx(output_hough, cv.MORPH_OPEN, kernel_hough_open)

    kernel_dilate = np.ones((25,25), np.uint8)
    hough_dilate_img = cv.morphologyEx(hough_open_img, cv.MORPH_DILATE, kernel_dilate)

    cv.imshow("test_hough", output_hough)
    cv.imshow("test_open", hough_open_img)
    cv.imshow("test_dilate", hough_dilate_img)

    '''

    
    # kernel = np.ones((5,5),np.uint8)
    # morph_close_img = cv.morphologyEx(laplacian_thresh, cv.MORPH_CLOSE, kernel)

    # kernel2 = np.ones((3,3),np.uint8)
    # morph_open_img = cv.morphologyEx(laplacian_thresh, cv.MORPH_OPEN, kernel2)
    
    # contours, hierarchy = cv.findContours(morph_open_img, mode = cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    # photo_contours = np.copy(photo)

    kernel = np.ones((21,21), np.uint8)
    test1 = cv.morphologyEx(laplacian_thresh, cv.MORPH_CROSS, kernel)

    # kernel2 = np.ones((7,7),np.uint8)
    # test2 = cv.morphologyEx(laplacian_thresh, cv.MORPH_CLOSE, kernel2)


    # cv.imshow("test1", test1)
    # cv.imshow("test2", test2)
    
    

    # Loop through each contour and assign a unique random BGR color
    # for i, cnt in enumerate(contours):
    #     color = np.random.randint(0, 256, size=3).tolist()  # Generates (B, G, R)
    #     cv.drawContours(photo_contours, contours, i, color, 2)


    # max_contour = max(contours, key=cv.contourArea)
    # x,y,w,h = cv.boundingRect(max_contour)
    # # draw the biggest contour (max_contour) in green
    # cv.rectangle(photo_contours,(x,y),(x+w,y+h),(0,255,0),2)
    
    
    

    # Calculate the circumference by counting all the non zero pixels in the image.
    #   This works since the above threshold function creates a strict black(255) white(0) picture.
    circumference = cv.countNonZero(laplacian_thresh)


    if ENABLE_VERBOSE:
        print("Circumference: {}".format(circumference))
        cv.waitKey(0)
   
    return circumference



def IC_pincount(photo: cv.typing.MatLike):
    CONTOUR_AREA_THRESH = 30            # This value was found to work best by trial and error, with a value of 50, the upright dip's pins get detected, but the upside down dips don't
    contour_count = 0

    # Create filter
    # (Filter values found by trial and error, optimizzed on DIP_8 pins)
    low_gray = np.array([50, 130, 220])
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



# Main program
if __name__ == "__main__":

    ##-- Processing --##
    # Create the necessary lists in the test_result_dict, every test needs a list.
    # In this list the quantified result of each test is saved.
    test_result_dict["circumference"] = []
    test_result_dict["IC_pincount"] = []

    # Grab filapaths for all photos going to be processed
    glob_filelist = glob.glob(TESTPHOTO_PATH + "/*.jpg")

    print("{} - Program started, analyzing {} photos".format(current_time(), len(glob_filelist)))

    # Loop through all photos and run processing.
    for file_path in glob_filelist:
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


        ##-- Parameter functions --##
        # test_result_dict["circumference"].append(circumference(photo))
        test_result_dict["IC_pincount"].append(IC_pincount(photo))


        ##-- Other test data (for plotting) --##
        test_result_dict["photo_number"].append(photo_count)                                # Save the current photo count in the same list position as the test results
        photo_count += 1                                                                    # Update the total photo_count
        test_result_dict["photo_component_name"].append(get_component_name(file_path))      # Save the current component name in the same list position as the test results

        # Give output to the user on every 500 photos analyzed. This gives the user insight in the programs speed.
        if photo_count % 500 == 0:
            print("Current photo count: {}/{}".format(photo_count, len(glob_filelist)))


    # Done processing
    print("{} - Done calculating. Total photos tested: {}".format(current_time(), photo_count))

    if ENABLE_VERBOSE: cv.destroyAllWindows()           # Sometimes the last window gets left behind, destroy all windows

    print("{} - Starting result visualisation".format(current_time()))


    ##-- Visualisation --##
    #### ---- IC_pincount ---- ####
    # START Gemini (AI) helped this bit, this generates our third scatter parameter, frequency of appearing and makes a new dictionary (grouped) that has the new information
    import pandas as pd

    # 1. Create DataFrame using only the two matching arrays
    df = pd.DataFrame({
        "photo_component_name": test_result_dict["photo_component_name"],
        "IC_pincount": test_result_dict["IC_pincount"]
    })

    # 2. Count frequencies of unique pairs
    grouped = df.groupby(["photo_component_name", "IC_pincount"]).size().reset_index(name="count")

    # 3. Scale frequency to marker size
    grouped["s"] = grouped["count"] * 20  # adjust factor as needed
    # END Gemini (AI) helped this bit



    # If this is done "axs." needs to be replaced with "axs[x]." with x representing the plot
    fig, axs = plt.subplots(1)

    # Plot the component name vs IC_pinout
    axs.scatter(grouped["photo_component_name"], grouped["IC_pincount"], s=grouped["s"], alpha=0.6)# Use a subplot for futureproofing, alowing for more plots in 1 window in the future.

    # axs.scatter(test_result_dict["photo_component_name"], test_result_dict["IC_pincount"], s=s)
    axs.set_title("IC_Pinout per component")
    axs.set(xlabel="Component", ylabel="IC_pincount (n)")
    axs.tick_params("x", labelrotation=45)

    #### ---- END IC_pincount ---- ####

    # Show the figures
    plt.show()



    print("{} - Program Done".format(current_time()))