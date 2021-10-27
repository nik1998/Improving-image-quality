import numpy as np
import cv2
from matplotlib import pyplot as plt
from mylibrary import *
from skimage.metrics import structural_similarity
from skimage import measure


def own_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


# Нерезкое маскирование
def unsharp_masking(image, sigma=2.0):
    # np.zeros(image.shape, dtype="float32")
    # smooth = cv2.GaussianBlur(image, (0, 0), 2)
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0)
    # return unsharp_image


def unknown_filter(image):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray = cv2.divide(image, bg, scale=255)
    return out_gray


'''
After greying the image try applying equalize histogram to the image, this allows the area's in the image with lower contrast to gain a higher contrast. 
Then blur the image to reduce the noise in the background. 
Next apply edge detection on the image, make sure that noise is sufficiently removed as ED is susceptible to it. 
Lastly, apply closing(dilation then erosion) on the image to close all the small holes inside the words.
'''


def custom_algorithm(img, sobel=False):
    showImage(img)
    # image = adaptive_hist(img, 10)
    image = cv2.GaussianBlur(img, (0, 0), 5.0)
    showImage(image)
    image = (255 * image).astype(np.uint8)
    if sobel:
        image = Sobel(image)
    else:
        image = cv2.Canny(image=image, threshold1=40, threshold2=60)
    image = image.astype(np.float) / 255
    showImage(image)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    res = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
    showImage(res)
    return res


def Sobel(image):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    image = cv2.filter2D(image, -1, kx)
    image = cv2.filter2D(image, -1, ky)
    # image = 255 - image
    return image


def Sobel_x(image):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    image = cv2.filter2D(image, -1, kx)
    return image


def Sobel_y(image):
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    image = cv2.filter2D(image, -1, ky)
    return image


def histogram_equalization(image):
    showImage(image)
    plt.figure()
    plt.hist((255 * image).ravel(), 256, [0, 255])
    hist, bins = np.histogram(image.ravel(), 256, range=[0, 1])
    cdf = hist.cumsum()
    plt.figure()
    plt.plot(range(256), cdf, 'r')
    # remap cdf to [0,1]
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])

    # generate img after Histogram Equalization
    img2 = (255 * image).astype(np.uint8)
    img2 = cdf[img2]
    #
    hist2, bins2 = np.histogram(img2.ravel(), 256, range=[0, 1])
    cdf2 = hist2.cumsum()
    plt.figure()
    plt.plot(hist2, 'g')
    plt.figure()
    plt.plot(range(256), cdf2, 'b')
    showImage(img2)
    # opencv realization
    # equ = cv2.equalizeHist((255 * image).astype(np.uint8))
    # equ = equ.astype(np.float) / 255
    # showImage(equ)
    plt.show()
    return img2.astype(np.float) / 255


def adaptive_hist(image, clipLimit=40):
    img = (image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    img = cl1.astype(np.float) / 255
    return img


def CLAHE_plot(image):
    pr = image
    res = []
    for i in range(1, 30):
        clahe = cv2.createCLAHE(clipLimit=i, tileGridSize=(8, 8))
        cl1 = clahe.apply(image)
        res.append(np.sum(np.abs(pr - cl1)))
        pr = cl1
    plt.plot(range(20, 30), res[19:])
    plt.show()


def CLAHE_experimental(image):
    h = image.shape[0]
    d = h // 8
    m = 0
    for i in range(0, h - d, d):
        for j in range(0, h - d, d):
            im = image[i:i + d, j:j + d]
            hist, bins = np.histogram(im.ravel(), 256, range=[0, 255])
            m = max(m, max(hist))
    return m


def gamma_correction(image, gamma=0.8):
    res = np.power(image, gamma)
    res = np.clip(res, 0.0, 1.0)
    showImage(res)
    return res


# http://amroamroamro.github.io/mexopencv/matlab/cv.PSNR.html
# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def psnr(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    hm = min(h1, h2)
    wm = min(w1, w2)
    psnr = cv2.PSNR(img1[:hm, :wm], img2[:hm, :wm], 1)
    return psnr


# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def similarity(img1, img2, show=False):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    hm = min(h1, h2)
    wm = min(w1, w2)
    (score, diff) = structural_similarity(img1[:hm, :wm], img2[:hm, :wm], full=True)
    if show:
        showImage(diff)
    return score


# https://cyberleninka.ru/article/n/mera-otsenki-rezkosti-tsifrovogo-izobrazheniya/viewer
def sharpness_measure(img):
    k = np.max(img) / 2
    n = img.shape[0]
    m = img.shape[1]
    d = (n - 1) * (m - 1)
    res = 0.0
    for i in range(1, n):
        for j in range(1, m):
            r = abs(img[i, j] - img[i, j - 1]) + abs(img[i, j] - img[i - 1, j])
            res += r * r
    return res / d / k


def contrast_measure(img):
    return img.std()


def compare_images_by_function(a, p, f, double=False):
    va = 0.0
    vp = 0.0
    for i, image in enumerate(a):
        if double:
            d = abs(f(image, p[i]))
            va += d
            vp += d
        else:
            va += f(image)
            vp += f(p[i])
    return va / len(a), vp / len(p)


def total_masking(image, sigma1=2.0, sigma2=2.0):
    img = cv2.GaussianBlur(image, (0, 0), sigma1)
    img = unsharp_masking(img, sigma2)
    img = unsharp_masking(img, sigma2)
    return np.clip(img, 0.0, 1.0)


def median_demo(target_array, array_length):
    sorted_array = np.sort(target_array)
    median = sorted_array[array_length // 2]
    return median


# https://github.com/sarnold/adaptive-median/blob/master/adaptive_median.py
def adaptive_median(image, window=3, threshold=0):
    ## set filter window and image dimensions
    filter_window = 2 * window + 1
    xlength, ylength = image.shape
    vlength = filter_window * filter_window

    ## create 2-D image array and initialize window
    image_array = image.copy()
    filter_window = np.zeros((filter_window, filter_window), dtype=np.float)
    target_vector = np.zeros(vlength, dtype=np.float)
    ## loop over image with specified window filter_window
    for y in range(window, ylength - (window + 1)):
        for x in range(window, xlength - (window + 1)):
            ## populate window, sort, find median
            filter_window = image[y - window:y + window + 1, x - window:x + window + 1]
            target_vector = np.reshape(filter_window, (vlength,))
            ## numpy sort
            median = median_demo(target_vector, vlength)
            ## check for threshold
            if threshold <= 0:
                image_array[y, x] = median
            else:
                scale = np.zeros(vlength)
                for n in range(vlength):
                    scale[n] = abs(target_vector[n] - median)
                scale = np.sort(scale)
                Sk = 1.4826 * scale[vlength // 2]
                if abs(image_array[y, x] - median) > threshold * Sk:
                    image_array[y, x] = median

    return image_array


def operate_binarization(img):
    img2 = img.copy()
    img2 = cv2.GaussianBlur(img2, (0, 0), 1)
    img2 = img2 > np.mean(img2)
    return np.concatenate((img, img2), axis=1)
