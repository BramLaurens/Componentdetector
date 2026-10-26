from calculate_parameters import calculate_parameters
import cv2 as cv
import pandas as pd
import sklearn as skl
from datetime import datetime

STREAM_URL = "rtsp://192.168.4.1/mjpeg/1"
DATA_PATH = "calculated_data.csv"
RANDOM_STATE = 42


random_forest = skl.ensemble.RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, max_depth=None)        # Default n_estimators and max_depth values


# A function that returns a string of formatted time.
def current_time():
    return datetime.now().strftime("%H:%M:%S.%f")

def train_model():
    print("{} - Loading training data from {}, training decision engine.".format(current_time(), DATA_PATH))

    data = pd.read_csv(DATA_PATH)
    x = data.drop(columns=['photo_number', 'component_name'])
    y = data['component_name']

    random_forest.fit(x, y)

    print("{} - Decision engine trained.".format(current_time()))


def draw_bounding_box(photo, label):
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

    export_photo = cv.rectangle(photo, (x_start_padding, y_start_padding), (x_end_padding, y_end_padding), (0, 255, 0), 2)
    export_photo = cv.putText(export_photo, label, (x_start_padding, y_start_padding - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return export_photo


if __name__ == "__main__":
    train_model()
    
    cap = cv.VideoCapture(STREAM_URL)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    while(1):
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        h, w, c = frame.shape
        frame = frame[0:h, 160:w-200]

        frame_parameters = calculate_parameters(frame)
        print(frame_parameters)

        prediction = random_forest.predict(frame_parameters)

        cv.imshow('frame', draw_bounding_box(frame, prediction[0]))

        
        print("{} - Predicted component: {}".format(current_time(), prediction))
        
        if cv.waitKey(1) == ord('q'):
            break