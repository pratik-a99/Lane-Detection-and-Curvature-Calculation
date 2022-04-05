__author__ = "Pratik Acharya"
__copyright__ = "Copyright 2022"
__UID__ = "117513615"

''' 

Instructions to run the code 
1. Make sure the images are in the same folder as the code and are named as given below
2. Run 'python Problem_1.py' in an ubuntu terminal

'''

import numpy as np
import cv2 as cv


def hist_eq(img_arr, hist):
    img_ret = np.zeros((img_arr.shape[0], img_arr.shape[1], 1), dtype="uint8")
    for i in range(img_arr.shape[1]):
        for j in range(img_arr.shape[0]):
            img_ret[j][i] = 255 * hist[img_arr[j][i]]

    return img_ret


def hist_cummulative(img_arr, clipping=False, clip_at=20):
    cumulative = np.zeros(256, np.int32)
    bins = np.zeros(256, np.int32)
    for i in range(img_arr.shape[1]):
        for j in range(img_arr.shape[0]):
            bins[img_arr[j][i]] += 1

    if clipping:
        extra = sum(bins[bins > clip_at] - clip_at)
        adding_extra = extra / 256

        bins = bins + adding_extra

    for i in range(len(bins)):
        cumulative[i] = sum(bins[0:i])

    return cumulative / sum(bins)


def hist_equalization(input_img, clipping=False, clip_at=20):
    input_img = cv.cvtColor(input_img, cv.COLOR_BGR2HSV)
    HChannel = input_img[:, :, 0]
    SChannel = input_img[:, :, 1]
    VChannel = input_img[:, :, 2]

    cumulative = hist_cummulative(VChannel, clipping, clip_at)
    new_V = hist_eq(VChannel, cumulative)

    hist_equalized = np.dstack((HChannel, SChannel, new_V))
    hist_equalized = cv.cvtColor(hist_equalized, cv.COLOR_HSV2BGR)

    return hist_equalized


def adaptive_hist(input_img, grid_x, grid_y, clipping=True, limit=20):
    adapted_img = input_img.copy()
    for i_part in range(int(input_img.shape[0] / grid_y)):
        for j_part in range(int(input_img.shape[1] / grid_x)):
            temp_img = input_img[i_part * grid_y: (i_part + 1) * grid_y,
                                 j_part * grid_x: (j_part + 1) * grid_x].copy()

            equalized = hist_equalization(temp_img, clipping, limit)
            adapted_img[i_part * grid_y: (i_part + 1) * grid_y,
                        j_part * grid_x: (j_part + 1) * grid_x] = equalized

    adapted_img = cv.bilateralFilter(adapted_img, 9, 20, 20)

    return adapted_img


if __name__ == '__main__':
    result_vid = cv.VideoWriter('HistogramEqualizationResult.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20.0, (1224, 370))
    result1 = cv.VideoWriter('AdaptiveHistogramResult.mp4', cv.VideoWriter_fourcc(*'mp4v'), 20.0, (1224, 370))
    for i in range(25):
        img = cv.imread('adaptive_hist_data/{:010d}.png'.format(i))

        result = hist_equalization(img)

        adaptive = adaptive_hist(img, 50, 50, clipping=True)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        cv.imshow('Original Image', img)
        cv.imshow('Histogram Equalization', result)
        result_vid.write(result)
        cv.imshow('Adaptive Histogram', adaptive)
        result1.write(adaptive)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
