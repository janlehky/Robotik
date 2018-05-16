import cv2
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering, AgglomerativeClustering


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
    top_left = [cols*0.1, rows*0.45]    # [cols*0.3, rows*0.6]
    bottom_right = [cols*0.99, rows*0.8]  # [cols*0.9, rows*0.95]
    top_right = [cols*0.9, rows*0.45]   # [cols*0.6, rows*0.6]

    verticles = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    return filter_region(image, verticles)


def calculate_points(polynom_par, width):
    x_points = [0, width*0.2, width*0.4, width*0.6, width*0.8, width]
    points = []
    p = np.poly1d(polynom_par)
    print("Polynom: {}".format(polynom_par))
    for x_point in x_points:
        y = p(x_point)
        X = [x_point, y]
        points.append(tuple(np.int32(X)))

    return points


img = cv2.imread('../images/cam.jpg', 1)

small = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)

gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

kernel_size = 15
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

edges = cv2.Canny(blurred, 45, 100) # (blurred, 50, 150)

area = select_region(edges)

lines = cv2.HoughLinesP(area, rho=1, theta=np.pi/180, threshold=10, minLineLength=10, maxLineGap=200)
# lines = cv2.HoughLinesP(area, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)

# try:
all_points = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    all_points.append([x1, y1])
    all_points.append([x2, y2])

X = np.array(all_points)

clstr = SpectralClustering(n_clusters=3, eigen_solver="arpack", affinity="nearest_neighbors")
result = clstr.fit_predict(X)

# connectivity matrix for structured Ward
connectivity = kneighbors_graph(
    X, n_neighbors=2, include_self=False)
# make connectivity symmetric
connectivity = 0.5 * (connectivity + connectivity.T)

ward = AgglomerativeClustering(n_clusters=2, linkage='ward', connectivity=connectivity)
result = ward.fit_predict(X)

points = zip(X, result)

print("Clustering: {}".format(result))

left_x_points = []
left_y_points = []
right_x_points = []
right_y_points = []
left_points = []
right_points = []

for point in points:
    X, y = point
    # print("X: {} y: {}".format(X, y))
    if y == 0:
        cv2.circle(small, tuple(X), radius=2, color=(255, 0, 0), thickness=2, lineType=8, shift=0)
        left_x_points.append(X[0])
        left_y_points.append(X[1])
        left_points.append(tuple(np.int32(X)))
    elif y == 1:
        cv2.circle(small, tuple(X), radius=2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)
        right_x_points.append(X[0])
        right_y_points.append(X[1])
        right_points.append(tuple(np.int32(X)))
    elif y == 2:
        cv2.circle(small, tuple(X), radius=2, color=(0, 0, 255), thickness=2, lineType=8, shift=0)
    # for line in lines:
    #     # print(line[0])
    #     x1, y1, x2, y2 = line[0]
    #     # print("x:{}".format(x1))
    #     X = (x1, y1)
    #     Y = (x2, y2)
    #     # print("x:{} y:{}".format(X, Y))
    #     # cv2.line(small, X, Y, (0, 255, 0), 2)
    #     cv2.circle(small, X, radius=2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)
    #     cv2.circle(small, Y, radius=2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)
left_line = np.polyfit(left_x_points, left_y_points, 2)
right_line = np.polyfit(right_x_points, right_y_points, 2)

lp = calculate_points(left_line, small.shape[1])
rp = calculate_points(right_line, small.shape[1])

lp = np.array(lp, dtype=np.int32)
rp = np.array(rp, dtype=np.int32)

print("Left Points: {}".format(lp))
cv2.polylines(small, [lp], isClosed=False, color=(255, 255, 0))
print("Right Points: {}".format(rp))
cv2.polylines(small, [rp], isClosed=False, color=(0, 255, 255))
# except:
#     pass

cv2.imshow('image', small)
cv2.waitKey(0)
cv2.destroyAllWindows()
