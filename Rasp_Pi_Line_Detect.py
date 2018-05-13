# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np


def filter_region(image, verticles):
    """

    :param image:
    :param verticles:
    :return:
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, verticles, 255)
    else:
        cv2.fillPoly(mask, verticles, (255, )*mask.shape[2])

    return cv2.bitwise_and(image, mask)


def select_region(image):
    """

    :param image:
    :return:
    """

    rows, cols = image.shape[:2]
    bottom_left = [cols*0.0, rows*0.8]  # [cols*0.1, rows*0.95]
    top_left = [cols*0.1, rows*0.2]    # [cols*0.3, rows*0.6]
    bottom_right = [cols*0.99, rows*0.8]  # [cols*0.9, rows*0.95]
    top_right = [cols*0.9, rows*0.2]   # [cols*0.6, rows*0.6]

    verticles = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    return filter_region(image, verticles)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    left_y = 0  # bottom point of left lines
    right_y = 0  # bottom point of right lines

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # ignore a vertical line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:  # y is reversed in image
                left_lines.append((slope, intercept))
                left_weights.append((length))
                if y1 > y2 and y1 > left_y:
                    left_y = y1
                elif y1 < y2 and y2 > left_y:
                    left_y = y2
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))
                if y1 > y2 and y1 > right_y:
                    right_y = y1
                elif y1 < y2 and y2 > right_y:
                    right_y = y2

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane, left_y, right_y  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line, line_prev):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line
    slope_prev, intercept_prev = line_prev

    slope = (slope + slope_prev)/2
    intercept = (intercept + intercept_prev)/2

    ALOT = 1e6

    # make sure everything is integer as cv2.line requires it
    x1 = max(min((y1 - intercept) / slope, ALOT), -ALOT)  # limit values before we try to convert them to integer
    x1 = int(x1)
    x2 = max(min((y2 - intercept) / slope, ALOT), -ALOT)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))
    
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# allow the camera to warmup
time.sleep(0.1)

left_line_prev = np.array([0, 0])
right_line_prev = np.array([0, 0])

# capture frames from the camera
for img in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = img.array

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 15
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    edges = cv2.Canny(blurred, 45, 100) # (blurred, 50, 150)

    area = select_region(edges)

    lines = cv2.HoughLinesP(area, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)
    # lines = cv2.HoughLinesP(area, rho=1, theta=np.pi / 180, threshold=30, minLineLength=20, maxLineGap=200)

    if lines is not None:
        left_lane, right_lane, left_y1, right_y1 = average_slope_intercept(lines)

        if left_lane is not None:
            np_left_lane = np.array(left_lane)
            left_line_prev = (np_left_lane + left_line_prev)/2

        left_lane = left_line_prev.tolist()

        if right_lane is not None:
            np_right_lane = np.array(right_lane)
            right_line_prev = (np_right_lane + right_line_prev) / 2

        right_lane = right_line_prev.tolist()

        # y1 = gray.shape[0]  # bottom of the image
        y2 = gray.shape[0] * 0.4    # top point y coordinate for line

        left_line = make_line_points(left_y1, y2, left_lane, left_line_prev)
        right_line = make_line_points(right_y1, y2, right_lane, right_line_prev)

        lines_final = [left_line, right_line]

        try:
            for line in lines_final:
                #print(line[0])
                # x1, y1, x2, y2 = line[0]
                # #print("x:{}".format(x1))
                # X = (x1, y1)
                # Y = (x2, y2)
                X, Y = line
                print("x:{} y:{}".format(X, Y))
                cv2.line(frame, X, Y, (0, 255, 0), 2)
        except:
            pass

    # show the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
