import numpy as np
import cv2


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
    bottom_left = [cols * 0.0, rows * 0.8]  # [cols*0.1, rows*0.95]
    top_left = [cols * 0.1, rows * 0.2]  # [cols*0.3, rows*0.6]
    bottom_right = [cols * 0.99, rows * 0.8]  # [cols*0.9, rows*0.95]
    top_right = [cols * 0.9, rows * 0.2]  # [cols*0.6, rows*0.6]

    verticles = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    return filter_region(image, verticles)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    left_y = 0     # bottom point of left lines
    right_y = 0    # bottom point of right lines

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


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    ALOT = 1e6

    # make sure everything is integer as cv2.line requires it
    x1 = max(min((y1 - intercept) / slope, ALOT), -ALOT)  # limit values before we try to convert them to integer
    x1 = int(x1)
    x2 = max(min((y2 - intercept) / slope, ALOT), -ALOT)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


img = cv2.imread('images/street_7.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel_size = 9
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

edges = cv2.Canny(blurred, 50, 150)

area = select_region(edges)

lines = cv2.HoughLinesP(area, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)

try:
    left_lane, right_lane, left_y1, right_y1 = average_slope_intercept(lines)

    y1 = img.shape[0]  # bottom of the image
    print(y1)
    y2 = y1 * 0.2  # slightly lower than the middle

    left_line = make_line_points(left_y1, y2, left_lane)
    right_line = make_line_points(right_y1, y2, right_lane)

    print(left_line)
    print(right_line)

    lines = [left_line, right_line]

    for line in lines:
        X, Y = line
        cv2.line(img, X, Y, (0, 255, 0), 2)
except:
    pass


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
