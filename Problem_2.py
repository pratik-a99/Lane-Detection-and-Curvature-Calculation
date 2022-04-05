__author__ = "Pratik Acharya"
__copyright__ = "Copyright 2022"
__UID__ = "117513615"

''' 

Instructions to run the code 
1. Make sure the video is in the same folder as the code and are named as given below
2. Run 'python Problem_2.py' in an ubuntu terminal

'''

import numpy as np
import cv2 as cv

cap = cv.VideoCapture("whiteline.mp4")
result = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20.0, (960, 540))


def lanes_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    canny_img = cv.Canny(blurred, 50, 150)

    height, width = gray.shape
    triangle = np.array([[(0, height), (int(width / 2), int(height / 2) + 45), (width, height)]])
    mask = np.zeros(gray.shape, dtype="uint8")
    mask = cv.fillPoly(mask, triangle, 255)
    mask = cv.bitwise_and(canny_img, mask)

    lines = cv.HoughLinesP(mask, 2, np.pi / 180, 5, np.array([]), 40, 30)

    left = []
    right = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            if fit[0] < 0:
                left.append((fit[0], fit[1]))
            else:
                right.append((fit[0], fit[1]))

    right_avg_line = np.average(right, axis=0)
    left_avg_line = np.average(left, axis=0)

    right_points = np.array([int((image.shape[0] - right_avg_line[1]) / right_avg_line[0]),
                             image.shape[0],
                             int((image.shape[0] * 0.62 - right_avg_line[1]) / right_avg_line[0]),
                             int(image.shape[0] * 0.62)])

    left_points = np.array([int((image.shape[0] - left_avg_line[1]) / left_avg_line[0]),
                            image.shape[0],
                            int((image.shape[0] * 0.62 - left_avg_line[1]) / left_avg_line[0]),
                            int(image.shape[0] * 0.62)])

    return np.array([left_points, right_points]), canny_img


def lane_type(canny_ip, line):
    height, width = canny_ip.shape

    x1, y1, x2, y2 = line
    rect = np.array([[(x1 + 15, y1), (x1 - 15, y1), (x2 + 10, y2 + 10), (x2 - 10, y2 - 10)]], dtype=np.int32)

    mask = np.zeros((height, width, 1), dtype="uint8")
    cv.fillPoly(mask, rect, 255)

    mask = cv.bitwise_and(canny_ip, mask)

    white_px = np.size(mask[mask == 255])

    return white_px

    # if white_px > 400:
    #     return False
    # else:
    #     return True


def display_lines(image, lines, dashed):
    lines_image = image.copy()
    if lines is not None:
        x1, y1, x2, y2 = lines
        if dashed:
            cv.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
        else:
            cv.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_image


if __name__ == '__main__':
    while cap.isOpened():
        ret, frame = cap.read()

        # To test on flipped video
        # frame = cv.flip(frame, 1)

        if ret:
            [points_left, points_right], canny = lanes_detection(frame)

            num_dotted_l = lane_type(canny, points_left)
            num_dotted_r = lane_type(canny, points_right)

            if num_dotted_r < num_dotted_l:
                dotted_r = True
                dotted_l = False
            else:
                dotted_r = False
                dotted_l = True

            frame = display_lines(frame, points_left, dotted_l)
            frame = display_lines(frame, points_right, dotted_r)

            cv.imshow('Video', frame)
            # result.write(frame)

            if cv.waitKey(25) & 0xFF == ord('q'):
                # result.release()
                break

        else:
            # result.release()
            break

    cv.destroyAllWindows()
    cap.release()
