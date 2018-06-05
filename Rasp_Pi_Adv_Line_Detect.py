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


def separate_lab(rgb_img):
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    l = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    return l, a, b


def seperate_hls(rgb_img):
    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]
    return h, l, s


def binary_threshold_lab_luv(rgb_img, athresh):
    l, a, b = separate_lab(rgb_img)
    binary = np.ones_like(l)
    binary[
        (a > athresh[0]) & (a <= athresh[1])
    ] = 0
    return binary


def gradient_threshold(channel, thresh):
    # Take the derivative in x
    sobelx = cv2.Sobel(channel, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold gradient channel
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return sxbinary


def histo_peak(histo):
    """Find left and right peaks of histogram"""
    midpoint = np.int(histo.shape[0]/2)
    leftx_base = np.argmax(histo[:midpoint])
    rightx_base = np.argmax(histo[midpoint:]) + midpoint
    return leftx_base, rightx_base


def get_lane_indices_sliding_windows(binary_warped, leftx_base, rightx_base, n_windows, margin, recenter_minpix):
    """Get lane line pixel indices by using sliding window technique"""
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    out_img = out_img.copy()
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / n_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > recenter_minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > recenter_minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    return left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img


def get_lines(img):

    A_CHANNEL_THRESH = (120, 255)
    GRADIENT_THRESH = (10, 50)

    # LAB and LUV channel threshold
    s_binary = binary_threshold_lab_luv(img, A_CHANNEL_THRESH)

    # Gradient threshold on S channel
    h, l, s = seperate_hls(img)
    sxbinary = gradient_threshold(h, GRADIENT_THRESH)

    # Combine two binary images to view their contribution in green and red
    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary))) * 255

    IMG_SIZE = img.shape[::-1][1:]
    OFFSET = 300

    PRES_SRC_PNTS = np.float32([
        (IMG_SIZE[0] * 0.02, IMG_SIZE[1] * 0.4),  # Top-left corner
        (1, IMG_SIZE[1]),  # Bottom-left corner
        (IMG_SIZE[0], IMG_SIZE[1]),  # Bottom-right corner
        (IMG_SIZE[0] * 0.97, IMG_SIZE[1] * 0.4)  # Top-right corner
    ])

    PRES_DST_PNTS = np.float32([
        [OFFSET, 0],
        [OFFSET, IMG_SIZE[1]],
        [IMG_SIZE[0] - OFFSET, IMG_SIZE[1]],
        [IMG_SIZE[0] - OFFSET, 0]
    ])

    M = cv2.getPerspectiveTransform(PRES_SRC_PNTS, PRES_DST_PNTS)
    M_INV = cv2.getPerspectiveTransform(PRES_DST_PNTS, PRES_SRC_PNTS)
    warped = cv2.warpPerspective(img, M, IMG_SIZE, flags=cv2.INTER_LINEAR)

    N_WINDOWS = 50
    MARGIN = 300
    RECENTER_MINPIX = 50

    # Warp binary image of lane line
    binary_warped = cv2.warpPerspective(s_binary, M, IMG_SIZE, flags=cv2.INTER_LINEAR)

    # Calculate histogram of lane line pixels
    histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)

    leftx_base, rightx_base = histo_peak(histogram)
    left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img = get_lane_indices_sliding_windows(
        binary_warped, leftx_base, rightx_base, N_WINDOWS, MARGIN, RECENTER_MINPIX)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return binary_warped, ploty, left_fitx, right_fitx, M_INV, leftx, lefty, rightx, righty


def project_lane_line(original_image, binary_warped, ploty, left_fitx, right_fitx, m_inv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (original_image.shape[1], original_image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    return result


def calc_curvature(ploty, leftx, rightx, lefty, righty, unit="m"):
    """returns curvature in meters."""
    # Define conversions in x and y from pixels space to meters
    YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
    XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*YM_PER_PIX, leftx*XM_PER_PIX, 2)
    right_fit_cr = np.polyfit(righty*YM_PER_PIX, rightx*XM_PER_PIX, 2)
    # Calculate the new radii of curvature
    left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvem = ((1 + (2*right_fit_cr[0]*y_eval*YM_PER_PIX + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    return left_curvem, right_curvem


# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

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

    binary_warped, ploty, left_fitx, right_fitx, M_INV, leftx, lefty, rightx, righty = get_lines(frame)
    img_result = project_lane_line(frame, binary_warped, ploty, left_fitx, right_fitx, M_INV)
    left_curvem, right_curvem = calc_curvature(ploty, leftx, rightx, lefty, righty)

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
