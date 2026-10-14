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
import pandas as pd                 # Pandas is used to store the collected paramterdata in an csv file, so it visualisation can happen without all calculations.


ENABLE_VERBOSE = False
CSV_NAME = "calculated_data.csv"


# Variables 
TESTPHOTO_PATH = "photos/train_photos/DATASET_BLUE_21091417"               # Path where photos are found
EMPTYPHOTO_PATH = "emptyphotos"
EMPTYREFERENCE_PATH = "emptyphotos/empty_reference.jpg"  # Path where the empty reference photo is found. This photo is used to subtract the turntable from the image, so only the component remains.

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



def count_unique_colors(photo: cv.typing.MatLike):
    # To create a body mask, we use the s channel of the HSV colorspace
    photo_HSV = cv.cvtColor(photo, cv.COLOR_BGR2HSV_FULL)

    h, s, v = cv.split(photo_HSV)

    # Increase contrast by 1.1 to make some connectors more visible, and apply a threshold
    s = cv.convertScaleAbs(s, alpha=1.1, beta=0)                    
    ret, thresh = cv.threshold(s, 100, 255, cv.THRESH_BINARY)

    # Apply mask to image using bitwise AND
    body_masked = cv.bitwise_and(photo, photo, mask=thresh)

    # Reshape the body_masked image to a 2D array of shape (num_pixels, 3). We basically throw away the positional information and keep the colors only.
    # The reshape takes new dimensions as parameters, in which -1 lets numpy calculate this itself.
    colors_array = body_masked.reshape(-1, body_masked.shape[2])

    # Convert the MatLike into uint8 values for h, s, v. This is possible since the photo is flattened to a 2D array in the above line.
    b = colors_array[:, 0].astype(np.uint32)
    g = colors_array[:, 1].astype(np.uint32)
    r = colors_array[:, 2].astype(np.uint32)

    # Use bitwise operators to add h, s and v after eachother.
    # Since h, s, and v are arrays, color_ids is now a 1D array with colors for all pixels in the image (saved as a single number)'
    color_ids = ((b << 16) | (g << 8) | r)

    # Use numpy's array.unique() function to find all uniques in the color_ids array. 
    values, counts = np.unique(color_ids, return_counts=True)

    # Calculate area, minimum must be 1 to avoid devision by 0
    oppervlakte = max(1, cv.countNonZero(thresh))


    return(len(values) / oppervlakte)
    # return 1

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


def body_mask_color(img_bgr, debug=False):
    blur = cv.GaussianBlur(img_bgr, (15, 15), 0)
    img_hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    lower1, upper1 = np.array([0, 120, 60]), np.array([14, 255, 255])
    lower2, upper2 = np.array([165, 120, 60]), np.array([180, 255, 255])
    bodymask = cv.inRange(img_hsv, lower1, upper1) | cv.inRange(img_hsv, lower2, upper2)

    return bodymask

#Builds a median reference image from all empty images
def build_empty_reference():
    empty_images = []
    glob_empty_filelist = glob.glob(EMPTYPHOTO_PATH + "/*.jpg")
    for file_path in glob_empty_filelist:
        img = cv.imread(file_path)
        if img is not None:
            empty_images.append(img)

    stack = np.stack(empty_images, axis=0)
    reference = np.median(stack, axis=0).astype(np.uint8)
    cv.imwrite("emptyphotos/empty_reference.jpg", reference)
    return reference

#loads the empty reference image, if it does not exist, it builds it first
def load_empty_reference():
    if not os.path.exists(EMPTYREFERENCE_PATH):
        print("Empty reference image not found. Building empty reference...")
        build_empty_reference()

    img = cv.imread(EMPTYREFERENCE_PATH)
    h, w, c = img.shape
    img = img[0:h, 160:w-200]
    return img

# clacluate the differece between the current image and the empty reference image, returns a 2D array of the difference
def difference_from_empty(img):
    ref = load_empty_reference()
    a_blur = cv.GaussianBlur(img,(5,5), 0)
    b_blur = cv.GaussianBlur(ref,(5,5), 0)
    a = cv.cvtColor(a_blur, cv.COLOR_BGR2LAB).astype(np.float32)
    b = cv.cvtColor(b_blur, cv.COLOR_BGR2LAB).astype(np.float32)
    #subtract the two arrays
    d = a - b
    #subtract the median of the difference to compensate for lighting changes
    d-=np.median(d, axis=0)
    #return the squared sum of the difference, which is a 2D array of the difference (distance in lab space)
    return np.linalg.norm(d, axis=2)

def body_mask_diff(mask, lead_width = 6, min_area=300):
    #Create kernel for morphological operations, the size of the kernel is based on the lead width of the component
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (lead_width+3, lead_width+3))
    # Apply morphological operations to remove noise and fill holes in the mask
    body = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    body = cv.morphologyEx(body, cv.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv.connectedComponentsWithStats(body)
    if n <= 1:
        return np.zeros_like(body)
    # Find the largest connected component (the body of the component)
    largest = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    if stats[largest, cv.CC_STAT_AREA] < min_area:
        return np.zeros_like(body)
    return np.where(labels == largest, 255, 0).astype(np.uint8)

def body_mask_diff_pipeline(img):
    cv.imshow("Original", img)
    diff = difference_from_empty(img)
    cv.imshow("Difference", np.clip(diff*4, 0, 255).astype(np.uint8))
    fg = ((diff > 40).astype(np.uint8) * 255)
    cv.imshow("Foreground", fg)

    #crop out the turntable edges
    h, w = fg.shape
    fg = fg[0:h, 160:w-200]
    body = body_mask_diff(fg, lead_width=6, min_area=300)

    cv.imshow("Body", body)
    cv.waitKey(0)

def body_mask_HSV(photo: cv.typing.MatLike):
    # Convert photo to HSV and apply a blur prevent white spots in threshold
    photo_HSV = cv.cvtColor(photo, cv.COLOR_BGR2HSV)
    photo_HSV_blur = cv.GaussianBlur(photo_HSV, (15,15), 0)

    h, s, v = cv.split(photo_HSV_blur)

    s = cv.convertScaleAbs(s, alpha=1.1, beta=0)                    # Increase contrast by 1.1 to make some connectors more visible
    ret, thresh = cv.threshold(s, 100, 255, cv.THRESH_BINARY)

    body_masked = cv.bitwise_and(photo, photo, mask=thresh)

    cv.imshow("mask", body_masked)

    return body_masked

def frequency_domain_test(photo: cv.typing.MatLike):
    photo_gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)

    # 1. Grayscale FFT
    photo_fft = np.fft.fft2(photo_gray)
    photo_fft_shift = np.fft.fftshift(photo_fft)

    # Calculate magnitude spectrum for grayscale (log scale to make frequencies visible)
    magnitude_gray = 20 * np.log(np.abs(photo_fft_shift) + 1)
    # Normalize to [0, 255] for OpenCV rendering
    magnitude_gray = cv.normalize(magnitude_gray, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    ret, magnitude_thresh = cv.threshold(magnitude_gray, 150, 255, cv.THRESH_BINARY)

    # fft_BGR = cv.cvtColor(magnitude_thresh, cv.COLOR_GRAY2BGR)



    lines = cv.HoughLinesP(
        magnitude_thresh, 
        rho=1, 
        theta=np.pi / 180, 
        threshold=20, 
        minLineLength=30, 
        maxLineGap=40
    )


    if ENABLE_VERBOSE:

        # Draw hough lines
        if lines is not None:
            for line in lines:
                # Flatten array to handle shape variations safely
                line_data = line.ravel()
                
                if len(line_data) == 4:
                    # Probabilistic Hough result: (x1, y1, x2, y2)
                    x1, y1, x2, y2 = line_data
                    cv.line(magnitude_gray, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    
                elif len(line_data) == 2:
                    # Standard Hough fallback: (rho, theta)
                    rho, theta = line_data
                    a, b = np.cos(theta), np.sin(theta)
                    x0, y0 = a * rho, b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv.line(magnitude_gray, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv.imshow('Grayscale FFT Magnitude', magnitude_gray)
        cv.imshow('Grayscale Image', photo_gray)

        cv.waitKey(0)

    total_lines = 0
    if lines is not None:
        total_lines = len(lines)
    else:
        total_lines = 0

    fft_opp = 0
    opp = cv.countNonZero(magnitude_thresh)
    if opp is not None:
        fft_opp = opp
    else:
        fft_opp = 0

    return {
        "fft_lines": total_lines,
        "fft_opp": fft_opp
        }

def calculate_fill_factor(photo: cv.typing.MatLike):
    diff = difference_from_empty(photo)
    masked_photo = ((diff > 40).astype(np.uint8) * 255)

    points = cv.findNonZero(masked_photo)       # Find non zero points and saave the coordinates of those.
    convex_hull = cv.convexHull(points)         # Calculate the convex hull around those points

    mask_area = cv.countNonZero(masked_photo)           # Calculate the area of the mask
    convex_hull_area = cv.contourArea(convex_hull)      # Calculate the area of the convex hull

    if mask_area == 0:
        fill_factor = 0     # prevent DIV0 error
    else:
        fill_factor =  mask_area / convex_hull_area


    if ENABLE_VERBOSE:
        masked_photo_copy = cv.cvtColor(masked_photo, cv.COLOR_GRAY2BGR)
        cv.drawContours(masked_photo_copy, [convex_hull], 0, (0, 0, 255), 2)
        print("mask area: \t\t\t{}".format(mask_area))
        print("Convex hull area: \t{}".format(convex_hull_area))
        print("area difference\t {}".format(fill_factor))

        # cv.imshow("Original", photo)
        # cv.imshow("Difference", np.clip(diff*4, 0, 255).astype(np.uint8))
        cv.imshow("fg", masked_photo)
        cv.imshow("convex_hull", masked_photo_copy)
        cv.waitKey(0)

    return fill_factor

# Main program
if __name__ == "__main__":


    ##-- Processing --##
    # Create the necessary lists in the test_result_dict, every test needs a list.
    # In this list the quantified result of each test is saved.
    test_result_dict["circumference"] = []
    test_result_dict["IC_pincount"] = []
    test_result_dict["unique_colors"] = []
    test_result_dict["fft_lines"] = []
    test_result_dict["fft_opp"] = []
    test_result_dict["fill_factor"] = []

    # Grab filapaths for all photos going to be processed
    glob_filelist = glob.glob(TESTPHOTO_PATH + "/*.jpg")

    print("{} - Program started, analyzing {} photos".format(current_time(), len(glob_filelist)))

    # Loop through all photos and run processing.
    for file_path in glob_filelist:
        # Read image
        photo = cv.imread(file_path, cv.IMREAD_COLOR_BGR)

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
        # rubberband(photo)

        ##-- Parameter functions --##
        # test_result_dict["circumference"].append(circumference(photo))
        # test_result_dict["IC_pincount"].append(IC_pincount(photo))
        # test_result_dict["circumference"].append(circumference(photo))
        test_result_dict["unique_colors"].append(count_unique_colors(photo_cropped))
        test_result_dict["fill_factor"].append(calculate_fill_factor(photo))

        fft_test = frequency_domain_test(photo_cropped)

        test_result_dict["fft_lines"].append(fft_test["fft_lines"])

        ##-- Other test data (for plotting) --##
        test_result_dict["photo_number"].append(photo_count)                                # Save the current photo count in the same list position as the test results
        photo_count += 1                                                                    # Update the total photo_count
        test_result_dict["photo_component_name"].append(get_component_name(file_path))      # Save the current component name in the same list position as the test results


        # Give output to the user on every 100 photos analyzed. This gives the user insight in the programs speed.
        if photo_count % 100 == 0:
            print("{} Current photo count: {}/{}".format(current_time(), photo_count, len(glob_filelist)))


    # Done processing
    print("{} - Done calculating. Total photos tested: {}".format(current_time(), photo_count))

    if ENABLE_VERBOSE: cv.destroyAllWindows()           # Sometimes the last window gets left behind, destroy all windows

    calculated_data = pd.DataFrame(data = {"photo_num": test_result_dict["photo_number"],
                                          "component_name": test_result_dict["photo_component_name"],
                                        #   "circumference": test_result_dict["circumference"],
                                        #   "IC_pincount": test_result_dict["IC_pincount"],
                                          "unique_colors": test_result_dict["unique_colors"],
                                          "fft_lines": test_result_dict["fft_lines"],
                                          "fill_factor": test_result_dict["fill_factor"]
                                          }
                                  )

    calculated_data.to_csv(CSV_NAME, index=False)

    print("Saved values to csv file: {}, exiting.".format(CSV_NAME))