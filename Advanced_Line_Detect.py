import numpy as np
import cv2


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

    A_CHANNEL_THRESH = (115, 125) #120, 255
    GRADIENT_THRESH = (10, 50)

    # LAB and LUV channel threshold
    s_binary = binary_threshold_lab_luv(img, A_CHANNEL_THRESH)
    l, a, b = separate_lab(img)

    # Gradient threshold on S channel
    h, l, s = seperate_hls(img)
    sxbinary = gradient_threshold(h, GRADIENT_THRESH)

    # Combine two binary images to view their contribution in green and red
    color_binary = np.dstack((sxbinary, s_binary, np.zeros_like(sxbinary))) * 255

    IMG_SIZE = img.shape[::-1][1:]
    OFFSET = 300

    PRES_SRC_PNTS = np.float32([
        (IMG_SIZE[0] * 0.02, IMG_SIZE[1] * 0.3),  # Top-left corner
        (1, IMG_SIZE[1]),  # Bottom-left corner
        (IMG_SIZE[0], IMG_SIZE[1]),  # Bottom-right corner
        (IMG_SIZE[0] * 0.97, IMG_SIZE[1] * 0.3)  # Top-right corner
    ])

    PRES_DST_PNTS = np.float32([
        [OFFSET, 0],
        [OFFSET, IMG_SIZE[1]],
        [IMG_SIZE[0] - OFFSET, IMG_SIZE[1]],
        [IMG_SIZE[0] - OFFSET, 0]
    ])

    M = cv2.getPerspectiveTransform(PRES_SRC_PNTS, PRES_DST_PNTS)
    M_INV = cv2.getPerspectiveTransform(PRES_DST_PNTS, PRES_SRC_PNTS)

    N_WINDOWS = 30
    MARGIN = 100
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

    return binary_warped, ploty, left_fitx, right_fitx, M_INV, leftx, lefty, rightx, righty #binary_warped


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


def find_lines(frame):
    binary_warped, ploty, left_fitx, right_fitx, M_INV, leftx, lefty, rightx, righty = get_lines(frame)
    img_result = project_lane_line(frame, binary_warped, ploty, left_fitx, right_fitx, M_INV)
    left_curvem, right_curvem = calc_curvature(ploty, leftx, rightx, lefty, righty)

    return binary_warped, left_curvem, right_curvem
