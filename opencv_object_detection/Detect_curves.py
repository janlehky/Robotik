import cv2
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, Birch
import matplotlib.pyplot as plt


GRADIENT_THRESH = (20, 100)
S_CHANNEL_THRESH = (50, 255)
L_CHANNEL_THRESH = (50, 255)
B_CHANNEL_THRESH = (150, 200)
L2_CHANNEL_THRESH = (225, 255)


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


def calculate_points(polynom_par, width, height):
    x_points = [0, width*0.1, width*0.2, width*0.4, width*0.5, width*0.6, width*0.80, width*0.9, width*0.95, width]
    points = []
    p = np.poly1d(polynom_par)
    print("Polynom: {}".format(polynom_par))
    for x_point in x_points:
        y = p(x_point)

        if y > height*0.01:
            X = [x_point, y]
            points.append(tuple(np.int32(X)))

    return points


def calculate_curvature(poly_par, x):
    """"Calculate curvature at point x"""

    curvature = np.absolute(2*poly_par[0])/((1+(2*poly_par[0]*x+poly_par[1])**2)**1.5)

    return curvature


def separate_hls(rgb_img):
    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    l = hls[:,:,1]
    s = hls[:,:,2]
    return h, l, s


def separate_lab(rgb_img):
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    l = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]
    return l, a, b


def separate_luv(rgb_img):
    luv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2Luv)
    l = luv[:,:,0]
    u = luv[:,:,1]
    v = luv[:,:,2]
    return l, u, v


def binary_threshold_lab_luv(rgb_img, bthresh, lthresh):
    l, a, b = separate_lab(rgb_img)
    l2, u, v = separate_luv(rgb_img)
    binary = np.zeros_like(l)
    binary[
        ((b > bthresh[0]) & (b <= bthresh[1])) |
        ((l2 > lthresh[0]) & (l2 <= lthresh[1]))
    ] = 1
    return binary


def binary_threshold_hls(rgb_img, sthresh, lthresh):
    h, l, s = separate_hls(rgb_img)
    binary = np.zeros_like(h)
    binary[
        ((s > sthresh[0]) & (s <= sthresh[1])) &
        ((l > lthresh[0]) & (l <= lthresh[1]))
    ] = 1
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


def compute_perspective_transform(binary_image):
    # Define 4 source and 4 destination points = np.float32([[,],[,],[,],[,]])
    shape = binary_image.shape[::-1]  # (width,height)
    w = shape[0]
    h = shape[1]

    OFFSET = 300

    bottom_left = [w * 0.0, h * 0.9]
    top_left = [w * 0.1, h * 0.45]
    bottom_right = [w * 0.99, h * 0.9]
    top_right = [w * 0.9, h * 0.45]

    transform_src = np.float32([top_left, bottom_left, bottom_right, top_right])
    transform_dst = np.float32([[OFFSET, 0], [OFFSET, h], [w-OFFSET, h], [w-OFFSET, 0]])
    M = cv2.getPerspectiveTransform(transform_src, transform_dst)
    M_Reverse = cv2.getPerspectiveTransform(transform_dst, transform_src)
    return M, M_Reverse


def apply_perspective_transform(binary_image, M, plot=False):
    warped_image = cv2.warpPerspective(binary_image, M, (binary_image.shape[1], binary_image.shape[0]),
                                       flags=cv2.INTER_NEAREST)  # keep same size as input image
    if (plot):
        # Ploting both images Binary and Warped
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Binary/Undistorted and Tresholded')
        ax1.imshow(binary_image, cmap='gray')
        ax2.set_title('Binary/Undistorted and Warped Image')
        ax2.imshow(warped_image, cmap='gray')
        plt.show()

    return warped_image


img = cv2.imread('../images/cam.jpg', 1)

resize = 1.0
small = cv2.resize(img, (0, 0), fx=resize, fy=resize)

gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

kernel_size = 15
blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

edges = cv2.Canny(blurred, 45, 100) # (blurred, 50, 150)

# LAB and LUV channel threshold# LAB a
s_binary = binary_threshold_lab_luv(small, B_CHANNEL_THRESH, L2_CHANNEL_THRESH)

# Gradient threshold on S channel
h, l, s = separate_hls(small)
sxbinary = gradient_threshold(s, GRADIENT_THRESH)

# Combine two binary images to view their contribution in green and red
color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary))) * 255

edges_h = cv2.Canny(h, 65, 135)  # (blurred, 50, 150)

l_2, a_2, b_2 = separate_lab(small)
edges_b = cv2.Canny(b_2, 25, 45)  # (blurred, 50, 150)
l_3, u_3, v_3 = separate_luv(small)
edges_u = cv2.Canny(u_3, 15, 35)  # (blurred, 50, 150)

combine = (edges_h + edges_b + edges_u) / 3

# Draw figure for binary images
f, axarr = plt.subplots(1, 4)
f.set_size_inches(25, 8)
axarr[0].imshow(small)
axarr[1].imshow(edges_h)
axarr[2].imshow(edges_b)
axarr[3].imshow(combine)
# axarr[1].imshow(s_binary, cmap='gray')
# axarr[2].imshow(sxbinary, cmap='gray')
# axarr[3].imshow(color_binary)
axarr[0].set_title("Undistorted Image")
axarr[1].set_title("B/L Channel Binary")
axarr[2].set_title("Gradient Threshold S/L-Channel Binary")
axarr[3].set_title("Combined Binary")
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[3].axis('off')
plt.show()

M, M_Reverse = compute_perspective_transform(combine)
warped_image = apply_perspective_transform(combine, M, False)
IMG_SIZE = small.shape[::-1][1:]
warped = cv2.warpPerspective(small, M, IMG_SIZE, flags=cv2.INTER_LINEAR)
# # f, axarr = plt.subplots(1, 2)
# # f.set_size_inches(18, 5)
# # axarr[0].imshow(warped)
# # axarr[1].imshow(small)
# # plt.show()
#
# Calculate histogram of lane line pixels
histogram = np.sum(warped_image[int(warped_image.shape[0]/2):, :], axis=0)
#
# # Draw figure for warped binary and histogram
f, axarr = plt.subplots(1, 2)
f.set_size_inches(18, 5)
axarr[0].imshow(warped_image, cmap='gray')
axarr[1].plot(histogram)
axarr[0].set_title("Warped Binary Lane Line")
axarr[1].set_title("Histogram of Lane line Pixels")
plt.show()
#
# area = select_region(edges)
#
# lines = cv2.HoughLinesP(warped_image, rho=1, theta=np.pi/180, threshold=10, minLineLength=10, maxLineGap=200)
# # lines = cv2.HoughLinesP(area, rho=1, theta=np.pi/180, threshold=30, minLineLength=20, maxLineGap=200)
#
# # try:
# all_points = []
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     all_points.append([x1, y1])
#     all_points.append([x2, y2])
#
# X = np.array(all_points)
#
# clstr = SpectralClustering(n_clusters=2, eigen_solver="arpack", affinity="nearest_neighbors")
# result = clstr.fit_predict(X)
#
# # connectivity matrix for structured Ward
# connectivity = kneighbors_graph(
#     X, n_neighbors=2, include_self=False)
# # make connectivity symmetric
# connectivity = 0.5 * (connectivity + connectivity.T)
#
# ward = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='cityblock', connectivity=connectivity)
# result = ward.fit_predict(X)
#
# birch = Birch(n_clusters=2)
# result = birch.fit_predict(X)
#
# points = zip(X, result)
#
# print("Clustering: {}".format(result))
#
# left_x_points = []
# left_y_points = []
# right_x_points = []
# right_y_points = []
# left_points = []
# right_points = []
#
# for point in points:
#     X, y = point
#     # print("X: {} y: {}".format(X, y))
#     if y == 0:
#         cv2.circle(warped, tuple(X), radius=2, color=(255, 0, 0), thickness=2, lineType=8, shift=0)
#         left_x_points.append(X[0])
#         left_y_points.append(X[1])
#         left_points.append(tuple(np.int32(X)))
#     elif y == 1:
#         cv2.circle(warped, tuple(X), radius=2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)
#         right_x_points.append(X[0])
#         right_y_points.append(X[1])
#         right_points.append(tuple(np.int32(X)))
#     elif y == 2:
#         cv2.circle(warped, tuple(X), radius=2, color=(0, 0, 255), thickness=2, lineType=8, shift=0)
#     # for line in lines:
#     #     # print(line[0])
#     #     x1, y1, x2, y2 = line[0]
#     #     # print("x:{}".format(x1))
#     #     X = (x1, y1)
#     #     Y = (x2, y2)
#     #     # print("x:{} y:{}".format(X, Y))
#     #     # cv2.line(small, X, Y, (0, 255, 0), 2)
#     #     cv2.circle(small, X, radius=2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)
#     #     cv2.circle(small, Y, radius=2, color=(0, 255, 0), thickness=2, lineType=8, shift=0)
# left_line = np.polyfit(left_x_points, left_y_points, 2)
# right_line = np.polyfit(right_x_points, right_y_points, 2)
#
# curvature_l = calculate_curvature(left_line, warped.shape[0]*0.8)
# curvature_r = calculate_curvature(right_line, warped.shape[0]*0.8)
# print("Curvature l: {} r: {}".format(curvature_l, curvature_r))
#
# lp = calculate_points(left_line, warped.shape[1], warped.shape[0])
# rp = calculate_points(right_line, warped.shape[1], warped.shape[0])
#
# lp = np.array(lp, dtype=np.int32)
# rp = np.array(rp, dtype=np.int32)
#
# print("Left Points: {}".format(lp))
# cv2.polylines(warped, [lp], isClosed=False, color=(255, 255, 0))
# print("Right Points: {}".format(rp))
# cv2.polylines(warped, [rp], isClosed=False, color=(0, 255, 255))
# # except:
# #     pass
#
# re_warped_image = apply_perspective_transform(warped, M_Reverse, True)
#
# resize = 0.3
# small = cv2.resize(re_warped_image, (0, 0), fx=resize, fy=resize)
# cv2.imshow('image', small)
cv2.waitKey(0)
cv2.destroyAllWindows()
