# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import Adafruit_PCA9685
import multiprocessing
import RPi.GPIO as GPIO
from Distance import Distance
import math
from Advanced_Line_Detect import find_lines

GPIO.setmode(GPIO.BCM)  # Set GPIO pin numbering

# define values for communication between processes
error = multiprocessing.Value('d', 0.0)
front_distance_sensor_1 = multiprocessing.Value('d', 10.0)
front_distance_sensor_2 = multiprocessing.Value('d', 10.0)

max_speed = 3000     # Maximum speed of vehicle


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
    top_left = [cols*0.1, rows*0.4]    # [cols*0.3, rows*0.6]
    bottom_right = [cols*0.99, rows*0.8]  # [cols*0.9, rows*0.95]
    top_right = [cols*0.9, rows*0.4]   # [cols*0.6, rows*0.6]

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
    if math.isnan(x1):
        x1 = 0
    else:
        x1 = int(x1)
    x2 = max(min((y2 - intercept) / slope, ALOT), -ALOT)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def get_stearing_error(y2, left_line, right_line, mid):
    """
    Calculate middle point for car and error to let and right line
    """
    ALOT = 1e6

    slope, intercept = left_line
    if slope != 0:
        x_l = max(min((y2 - intercept) / slope, ALOT), -ALOT)
    else:
        x_l = 0

    slope, intercept = right_line
    if slope != 0:
        x_r = max(min((y2 - intercept) / slope, ALOT), -ALOT)
    else:
        x_r = 2*mid - 1

    left_err = mid - x_l
    right_err = x_r - mid
    print("l_err: {} r_err: {}".format(left_err, right_err))

    return ((left_err - right_err)/(left_err + right_err))


def calculate_speeds(error, front_distance_1, front_distance_2, max_speed):
    """
    Drive control logic
    """
    error_clamped = max(min(error, 1.0), -1.0)

    if front_distance_1 > 10 and front_distance_2 > 10:
        if error_clamped > 0:
            left_speed = (1 - error_clamped) * max_speed
            right_speed = max_speed
        elif error_clamped < 0:
            left_speed = max_speed
            right_speed = (1 + error_clamped) * max_speed
        else:
            left_speed = right_speed = max_speed
    else:
        left_speed = 0
        right_speed = 0

    return left_speed, right_speed


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (1600, 1200)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(1600, 1200))

# allow the camera to warmup
time.sleep(0.1)

enable_drives = False

if enable_drives:
    # Initialise the PCA9685 using the default address (0x40).
    pwm = Adafruit_PCA9685.PCA9685()

    # set number of pins for direction of drives
    left_fwd_pin_1 = 4
    left_fwd_pin_2 = 17
    left_bwd_pin_1 = 18
    left_bwd_pin_2 = 23

    right_fwd_pin_1 = 22
    right_fwd_pin_2 = 27
    right_bwd_pin_1 = 24
    right_bwd_pin_2 = 25

    GPIO.setup(left_fwd_pin_1, GPIO.OUT)  # left forward 1 pin
    GPIO.setup(left_fwd_pin_2, GPIO.OUT)  # left forward 2 pin
    GPIO.setup(left_bwd_pin_1, GPIO.OUT)  # left backward 1 pin
    GPIO.setup(left_bwd_pin_2, GPIO.OUT)  # left backward 2 pin

    GPIO.setup(right_fwd_pin_1, GPIO.OUT)  # right forward 1 pin
    GPIO.setup(right_fwd_pin_2, GPIO.OUT)  # right forward 2 pin
    GPIO.setup(right_bwd_pin_1, GPIO.OUT)  # right backward 1 pin
    GPIO.setup(right_bwd_pin_2, GPIO.OUT)  # right backward 2 pin

    left_fwd = True
    left_bwd = False

    right_fwd = True
    right_bwd = False

left_line_prev = np.array([0, 0])
right_line_prev = np.array([0, 0])

# Create measurement object for front sensor
front_measurement_1 = Distance(20, 21)
front_measurement_2 = Distance(26, 19)
# Create front measurement process and start it
front_measurement_process_1 = multiprocessing.Process(target=front_measurement_1.measure,
                                                      args=(front_distance_sensor_1, ))
front_measurement_process_1.start()

front_measurement_process_2 = multiprocessing.Process(target=front_measurement_2.measure,
                                                      args=(front_distance_sensor_2, ))
front_measurement_process_2.start()

# capture frames from the camera
for img in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    frame = img.array

    img_result, left_curvem, right_curvem = find_lines(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 15
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    edges = cv2.Canny(blurred, 45, 100) # (blurred, 50, 150)

    area = select_region(edges)

    lines = cv2.HoughLinesP(area, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)

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
        
        err = get_stearing_error(y2, left_lane, right_lane, gray.shape[1]/2)
        print("Error: {} Front 1: {}".format(err, front_distance_sensor_1.value))
        
        if enable_drives:
            left_speed, right_speed = calculate_speeds(err, front_distance_sensor_1.value, front_distance_sensor_2.value, max_speed)
            print("Left: {} Right: {}".format(left_speed, right_speed))
            # Right drives
            pwm.set_pwm(0, 0, int(right_speed))
            pwm.set_pwm(1, 0, int(right_speed))
            GPIO.output(right_fwd_pin_1, right_fwd)
            GPIO.output(right_fwd_pin_2, right_fwd)
            GPIO.output(right_bwd_pin_1, right_bwd)
            GPIO.output(right_bwd_pin_2, right_bwd)
           
            # Left drives
            pwm.set_pwm(4, 0, int(left_speed))
            pwm.set_pwm(5, 0, int(left_speed))
            GPIO.output(left_fwd_pin_1, left_fwd)
            GPIO.output(left_fwd_pin_2, left_fwd)
            GPIO.output(left_bwd_pin_1, left_bwd)
            GPIO.output(left_bwd_pin_2, left_bwd)

        left_line = make_line_points(left_y1, y2, left_lane, left_line_prev)
        right_line = make_line_points(right_y1, y2, right_lane, right_line_prev)

        lines_final = [left_line, right_line]

        try:
            for line in lines:
                # print(line[0])
                x1, y1, x2, y2 = line[0]
                # print("x:{}".format(x1))
                X = (x1, y1)
                Y = (x2, y2)
                # X, Y = line
                # print("x:{} y:{}".format(X, Y))
                cv2.line(frame, X, Y, (0, 255, 0), 2)
        except:
            pass

    # show the frame
    # cv2.imshow("Frame", frame)
    cv2.imshow("Frame", img_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        front_measurement_process_1.terminate()
        front_measurement_1.cleanup()
        front_measurement_process_2.terminate()
        front_measurement_2.cleanup()
        break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
