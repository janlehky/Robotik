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
    bottom_left = [cols*0.1, rows*0.92]  # [cols*0.1, rows*0.95]
    top_left = [cols*0.35, rows*0.6]    # [cols*0.3, rows*0.6]
    bottom_right = [cols*0.9, rows*0.92]  # [cols*0.9, rows*0.95]
    top_right = [cols*0.65, rows*0.6]   # [cols*0.6, rows*0.6]

    verticles = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    return filter_region(image, verticles)


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

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
            else:
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


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


# cap = cv2.VideoCapture('Driving - 800.mp4')
cap = cv2.VideoCapture('Self Driving Car_ Vehicle Detection.mp4')
left_line_prev = np.array([0, 0])
right_line_prev = np.array([0, 0])

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kernel_size = 9
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        edges = cv2.Canny(blurred, 45, 100) # (blurred, 50, 150)

        area = select_region(edges)

        lines = cv2.HoughLinesP(area, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)
        # lines = cv2.HoughLinesP(area, rho=1, theta=np.pi / 180, threshold=30, minLineLength=20, maxLineGap=200)

        if lines is not None:
            left_lane, right_lane = average_slope_intercept(lines)

            if left_lane is not None:
                np_left_lane = np.array(left_lane)
                left_line_prev = (np_left_lane + left_line_prev)/2

            left_lane = left_line_prev.tolist()

            if right_lane is not None:
                np_right_lane = np.array(right_lane)
                right_line_prev = (np_right_lane + right_line_prev) / 2

            right_lane = right_line_prev.tolist()

            y1 = gray.shape[0]  # bottom of the image
            y2 = y1 * 0.6  # slightly lower than the middle

            left_line = make_line_points(y1, y2, left_lane, left_line_prev)
            right_line = make_line_points(y1, y2, right_lane, right_line_prev)

            # print(left_line)
            # print(right_line)

            lines = [left_line, right_line]

            try:
                for line in lines:
                    X, Y = line
                    cv2.line(frame, X, Y, (0, 255, 0), 2)
            except:
                pass

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
