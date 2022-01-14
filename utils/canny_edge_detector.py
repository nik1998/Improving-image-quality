# https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
from utils.img_filters import *
import numpy as np


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def sobel_filters(img):
    ix = sobel_x(img)
    iy = sobel_y(img)
    G = np.hypot(ix, iy)
    G = G / G.max() * 255
    theta = np.arctan2(iy, ix)
    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0
            except IndexError as e:
                pass
    return Z


def hysteresis(img, weak, strong):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass

    return img


def threshold(img, weak, strong, lowthreshold, highthreshold):
    highThreshold = img.max() * highthreshold
    lowThreshold = highThreshold * lowthreshold
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res


def detect(img, sigma=1, weak_pixel=75, strong_pixel=255, lowthreshold=0.05,
           highthreshold=0.15):
    img_smoothed = cv2.GaussianBlur(img, (0, 0), sigma)  # convolve(img, gaussian_kernel(kernel_size, sigma))
    gradient_mat, theta_mat = sobel_filters(img_smoothed)
    nonMaxImg = non_max_suppression(gradient_mat, theta_mat)
    thresholdImg = threshold(nonMaxImg, weak_pixel, strong_pixel, lowthreshold, highthreshold)
    img_final = hysteresis(thresholdImg, weak_pixel, strong_pixel)
    return img_final