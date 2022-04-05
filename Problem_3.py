__author__ = "Pratik Acharya"
__copyright__ = "Copyright 2022"
__UID__ = "117513615"

''' 

Instructions to run the code 
1. Make sure the video is in the same folder as the code and are named as given below
2. Run 'python Problem_3.py' in an ubuntu terminal

'''

import numpy as np
import cv2 as cv


def perspective_warp(img):
    height, width = img.shape[:2]
    src = np.float32([(int(width / 2) - 85, int(height / 2) + 85),
                      (int(width / 2) + 85, int(height / 2) + 85),
                      (100, height),
                      (width, height)])

    dst_width = 540
    dst_height = 720

    dst = np.float32([(0, 0), (dst_width, 0), (0, dst_height), (dst_width, dst_height)])

    M = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(img, M, (dst_width, dst_height))

    return warped


def inv_warp(img, orig_img):
    height, width = orig_img.shape[:2]

    dst = np.float32([(int(width / 2) - 85, int(height / 2) + 85),
                      (int(width / 2) + 85, int(height / 2) + 85),
                      (100, height),
                      (width, height)])

    src_width = 540
    src_height = 720

    src = np.float32([(0, 0), (src_width, 0), (0, src_height), (src_width, src_height)])

    M = cv.getPerspectiveTransform(src, dst)

    warped = cv.warpPerspective(img, M, (width, height))

    return warped


def hls_threshold(img_orig, low_thresh, upper_thresh):
    img = img_orig.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    mask = cv.inRange(img, low_thresh, upper_thresh)
    hls_th = cv.bitwise_and(img, img, mask=mask).astype(np.uint8)
    hls_th = cv.cvtColor(hls_th, cv.COLOR_HLS2RGB)
    hls_th = cv.cvtColor(hls_th, cv.COLOR_RGB2GRAY)
    _, hls_th = cv.threshold(hls_th, 20, 255, cv.THRESH_BINARY)

    return hls_th


def sliding_window(img, win_width, windows):
    # output = img.copy()
    points_l = np.empty((0, 2), dtype=np.int32)
    points_r = np.empty((0, 2), dtype=np.int32)

    win_height = img.shape[0] // windows

    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)

    left_max = np.argmax(hist[:len(hist) // 2])
    right_max = len(hist) // 2 + np.argmax(hist[(len(hist) // 2):])

    rect_left = np.array([[(left_max - win_width // 2, img.shape[0]),
                           (left_max + win_width // 2, img.shape[0]),
                           (left_max + win_width // 2, img.shape[0] - win_height),
                           (left_max - win_width // 2, img.shape[0] - win_height)]], dtype=np.int32)

    rect_right = np.array([[(right_max - win_width // 2, img.shape[0]),
                            (right_max + win_width // 2, img.shape[0]),
                            (right_max + win_width // 2, img.shape[0] - win_height),
                            (right_max - win_width // 2, img.shape[0] - win_height)]], dtype=np.int32)

    for i in range(windows):

        l_mask = np.zeros_like(img)
        r_mask = np.zeros_like(img)

        cv.fillPoly(l_mask, rect_left, 255)
        cv.fillPoly(r_mask, rect_right, 255)

        l_mask = cv.bitwise_and(img, l_mask)
        r_mask = cv.bitwise_and(img, r_mask)

        l_white_px = np.argwhere(l_mask == 255)
        r_white_px = np.argwhere(r_mask == 255)

        points_l = np.append(points_l, l_white_px)
        points_r = np.append(points_r, r_white_px)

        if l_white_px.shape[0] != 0:
            left_max = int(np.mean(l_white_px[:, 1]))
            rect_left = np.array([[(left_max - win_width // 2, img.shape[0] - win_height * (i + 1)),
                                   (left_max + win_width // 2, img.shape[0] - win_height * (i + 1)),
                                   (left_max + win_width // 2, img.shape[0] - win_height * (i + 2)),
                                   (left_max - win_width // 2, img.shape[0] - win_height * (i + 2))]], dtype=np.int32)
        else:
            rect_left = np.array([[(left_max - win_width // 2, img.shape[0] - win_height * (i + 1)),
                                   (left_max + win_width // 2, img.shape[0] - win_height * (i + 1)),
                                   (left_max + win_width // 2, img.shape[0] - win_height * (i + 2)),
                                   (left_max - win_width // 2, img.shape[0] - win_height * (i + 2))]], dtype=np.int32)

        if r_white_px.shape[0] != 0:
            right_max = int(np.mean(r_white_px[:, 1]))
            rect_right = np.array([[(right_max - win_width // 2, img.shape[0] - win_height * (i + 1)),
                                    (right_max + win_width // 2, img.shape[0] - win_height * (i + 1)),
                                    (right_max + win_width // 2, img.shape[0] - win_height * (i + 2)),
                                    (right_max - win_width // 2, img.shape[0] - win_height * (i + 2))]], dtype=np.int32)
        else:
            rect_right = np.array([[(right_max - win_width // 2, img.shape[0] - win_height * (i + 1)),
                                    (right_max + win_width // 2, img.shape[0] - win_height * (i + 1)),
                                    (right_max + win_width // 2, img.shape[0] - win_height * (i + 2)),
                                    (right_max - win_width // 2, img.shape[0] - win_height * (i + 2))]], dtype=np.int32)

    return points_l, points_r, img


def rad_o_curvature(image, fit):
    height = image.shape[0]

    y_per_pixel = 32 / height

    curve = ((1 + (2 * fit[0] * y_per_pixel + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    return curve


def curve_fitting(image, left_points, right_points):

    global old_left_fit, old_right_fit, old_left_fitx, old_right_fitx

    points_l_c = left_points.reshape(-1, 2)
    points_r_c = right_points.reshape(-1, 2)

    x_dump_c = np.asarray(list(range(0, roi.shape[0])))

    left_fit_c = old_left_fit
    right_fit_c = old_right_fit

    left_fitx_c = old_left_fitx
    right_fitx_c = old_right_fitx

    if len(points_l_c) != 0:
        left_fit_c = np.polyfit(points_l_c[:, 0], points_l_c[:, 1], 2)
        left_fitx_c = left_fit_c[0] * (x_dump_c ** 2) + left_fit_c[1] * x_dump_c + left_fit_c[2]

    if len(points_r_c) != 0:
        right_fit_c = np.polyfit(points_r_c[:, 0], points_r_c[:, 1], 2)
        right_fitx_c = right_fit_c[0] * (x_dump_c ** 2) + right_fit_c[1] * x_dump_c + right_fit_c[2]

    return left_fit_c, right_fit_c, left_fitx_c, right_fitx_c


def draw_lanes(image, left_fit_x, right_fit_x, only_lane=False):
    blank_image = np.zeros_like(image)
    turning = 'straight'

    x_dump_c = np.asarray(list(range(0, roi.shape[0])))

    if not only_lane:
        lane_l = np.vstack((left_fit_x, x_dump_c)).T
        lane_r = np.vstack((right_fit_x, x_dump_c)).T
        points_poly = np.int_(np.append(lane_r, lane_l[::-1], axis=0))
        cv.fillPoly(blank_image, [points_poly], (0, 0, 200))
        for arr in range(30, len(x_dump_c), 50):
            blank_image = cv.arrowedLine(blank_image,
                                         (int((left_fit_x[arr] + right_fit_x[arr]) // 2), int(x_dump_c[arr])),
                                         (int((left_fit_x[arr - 30] + right_fit_x[arr]) // 2), int(x_dump_c[arr - 30])),
                                         (255, 0, 0), 2)

        for jxx in range(len(x_dump_c)):
            cv.circle(blank_image, (int(left_fit_x[jxx]), int(x_dump_c[jxx])), 5, (0, 255, 0), -1)
            cv.circle(blank_image, (int(right_fit_x[jxx]), int(x_dump_c[jxx])), 5, (0, 255, 0), -1)

    else:
        for jxx in range(len(x_dump_c)):
            cv.circle(image, (int(left_fit_x[jxx]), int(x_dump_c[jxx])), 5, (0, 0, 255), -1)
            cv.circle(image, (int(right_fit_x[jxx]), int(x_dump_c[jxx])), 5, (0, 0, 255), -1)
            blank_image = image

    if ((left_fit_x[len(left_fit_x)//2] + right_fit_x[len(right_fit_x)//2]) // 2) > 10 + image.shape[1] // 2:
        turning = 'right'
    elif ((left_fit_x[len(left_fit_x)//2] + right_fit_x[len(right_fit_x)//2]) // 2) < 10 - image.shape[1] // 2:
        turning = 'left'

    return blank_image, turning


def combined_op(original_image, output_image, roi_image, top_view,
                poly_image, img_size, left_curve, right_curve, turning):
    blank_image = np.zeros((img_size[0], img_size[1], 3), np.uint8)

    output_image = cv.resize(output_image, (int(img_size[1] * 0.6), int(img_size[0] * 0.7)),
                             interpolation=cv.INTER_AREA)
    original_image = cv.resize(original_image, (int(img_size[1] * 0.2), int(img_size[0] * 0.2)),
                               interpolation=cv.INTER_AREA)
    roi_image = cv.resize(roi_image, (int(img_size[1] * 0.2), int(img_size[0] * 0.2)),
                          interpolation=cv.INTER_AREA)
    roi_image = cv.cvtColor(roi_image, cv.COLOR_GRAY2BGR)

    top_view = cv.resize(top_view, (int(img_size[1] * 0.2), int(img_size[0] * 0.5)),
                         interpolation=cv.INTER_AREA)
    top_view = cv.cvtColor(top_view, cv.COLOR_GRAY2BGR)

    poly_image = cv.resize(poly_image, (int(img_size[1] * 0.2), int(img_size[0] * 0.5)),
                           interpolation=cv.INTER_AREA)

    blank_image[0:output_image.shape[0], 0:output_image.shape[1]] = output_image

    blank_image[0:original_image.shape[0],
                output_image.shape[1]:output_image.shape[1] + original_image.shape[1]] = original_image
    blank_image[0:roi_image.shape[0],
                output_image.shape[1] + original_image.shape[1]:] = roi_image

    blank_image[original_image.shape[0]:original_image.shape[0]+top_view.shape[0],
                output_image.shape[1]:output_image.shape[1] + top_view.shape[1]] = top_view
    blank_image[original_image.shape[0]:original_image.shape[0] + poly_image.shape[0],
                output_image.shape[1] + top_view.shape[1]:] = poly_image

    font = cv.FONT_HERSHEY_SIMPLEX
    blank_image = cv.putText(blank_image, 'Left Curvature: ' + str(round(left_curve, 2)) + ',',
                             (10, output_image.shape[0]+50), font, 1, (255, 255, 255), 2, cv.LINE_AA)
    blank_image = cv.putText(blank_image, 'Right Curvature: ' + str(round(right_curve, 2)),
                             (450, output_image.shape[0] + 50), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    blank_image = cv.putText(blank_image, 'Average Curvature: ' + str(round((right_curve+left_curve)/2, 2)),
                             (10, output_image.shape[0] + 100), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    blank_image = cv.putText(blank_image, '(1)', (output_image.shape[1] + 5, 20),
                             font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    blank_image = cv.putText(blank_image, '(2)', (output_image.shape[1] + original_image.shape[1] + 5, 20),
                             font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    blank_image = cv.putText(blank_image, '(3)', (output_image.shape[1] + 5, original_image.shape[0] + 20),
                             font, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    blank_image = cv.putText(blank_image, '(4)', (output_image.shape[1] + original_image.shape[1] + 5,
                                                  original_image.shape[0] + 20),
                             font, 0.5, (255, 255, 255), 2, cv.LINE_AA)

    blank_image = cv.putText(blank_image, '(1) Original Image, (2) Detected Lanes, (3) Warped Top View, '
                                          '(4) Detected points and Curve Fitting',
                             (10, output_image.shape[0] + 150), font, 0.8, (255, 255, 255), 2, cv.LINE_AA)

    blank_image = cv.putText(blank_image, '[Press q to quit]',
                             (1250, output_image.shape[0] + 195), font, 0.5, (255, 255, 255), 2, cv.LINE_AA)

    if turning == 'right':
        blank_image = cv.putText(blank_image, 'Turn Right',
                                 (10, 25), font, 0.7, (0, 0, 255), 2, cv.LINE_AA)
    elif turning == 'left':
        blank_image = cv.putText(blank_image, 'Turn Left',
                                 (10, 25), font, 0.7, (0, 0, 255), 2, cv.LINE_AA)
    else:
        blank_image = cv.putText(blank_image, 'Go Straight',
                                 (10, 25), font, 0.7, (0, 0, 255), 2, cv.LINE_AA)

    return blank_image


def draw_detected(image, points_left, points_right):
    copy = image.copy()
    copy = cv.cvtColor(copy, cv.COLOR_GRAY2BGR)
    points_left = points_left.reshape(-1, 2)
    points_right = points_right.reshape(-1, 2)
    for i in range(len(points_left)):
        cv.circle(copy, (int(points_left[i][1]), int(points_left[i][0])), 10, (200, 100, 0), 1)
    for i in range(len(points_right)):
        cv.circle(copy, (int(points_right[i][1]), int(points_right[i][0])), 10, (100, 200, 0), 1)

    return copy


old_left_fitx = None
old_right_fitx = None
old_left_fit = None
old_right_fit = None


if __name__ == '__main__':

    cap = cv.VideoCapture("challenge.mp4")
    result = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20.0, (1400, 700))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            roi = perspective_warp(frame)

            lower = np.array([0, 200, 0], dtype="uint8")
            upper = np.array([255, 255, 255], dtype="uint8")
            hls_w = hls_threshold(roi, lower, upper)
            hls_orig_w = hls_threshold(frame, lower, upper)

            lower = np.array([20, 120, 80], dtype="uint8")
            upper = np.array([45, 200, 255], dtype="uint8")
            hls_y = hls_threshold(roi, lower, upper)
            hls_orig_y = hls_threshold(frame, lower, upper)

            lanes = cv.bitwise_or(hls_w, hls_y)
            orig = cv.bitwise_or(hls_orig_w, hls_orig_y)

            points_l_, points_r_, op = sliding_window(lanes, 200, 10)

            x_dump = np.asarray(list(range(0, roi.shape[0])))

            left_fit, right_fit, left_fitx, right_fitx = curve_fitting(roi, points_l_, points_r_)

            rad_o_curvature_l = rad_o_curvature(roi, left_fit)
            rad_o_curvature_r = rad_o_curvature(roi, right_fit)

            drawn_image, turn = draw_lanes(roi, left_fitx, right_fitx)

            orig_lanes = inv_warp(drawn_image, frame)

            output = cv.addWeighted(frame, 1, orig_lanes, 0.4, 0)

            detected_image = draw_detected(lanes, points_l_, points_r_)
            only_lanes, _ = draw_lanes(detected_image, left_fitx, right_fitx, only_lane=True)

            final_output = combined_op(frame, output, orig, lanes, only_lanes, (700, 1400),
                                       rad_o_curvature_l, rad_o_curvature_r, turn)

            cv.imshow("orig", final_output)
            # result.write(final_output)

            old_left_fitx = left_fitx
            old_right_fitx = right_fitx

            old_left_fit = left_fit
            old_right_fit = right_fit

            if cv.waitKey(25) & 0xFF == ord('q'):
                # result.release()
                break

        else:
            # result.release()
            break

    cv.destroyAllWindows()
    cap.release()
