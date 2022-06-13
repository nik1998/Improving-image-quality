import random
from typing import List

from utils.mylibrary import *


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
    gaussian_3 = np.zeros_like(image)
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), sigma, gaussian_3)
    return 1.5 * image - 0.5 * gaussian_3
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


def sobel_x(image):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    image = cv2.filter2D(image, -1, kx)
    return image


def sobel_y(image):
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
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


# https://www.pyimagesearch.com/2021/02/01/opencv-histogram-equalization-and-adaptive-histogram-equalization-clahe/
def adaptive_hist(image, clipLimit=5):
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


def total_masking(image, sigma1=2.0, sigma2=2.0):
    img = cv2.GaussianBlur(image, (0, 0), sigma1)
    img = unsharp_masking(img, sigma2)
    img = unsharp_masking(img, sigma2)
    return np.clip(img, 0.0, 1.0)


def adaptive_median(image, window=3, threshold=0):
    ## set filter window and image dimensions
    filter_window = 2 * window + 1
    xlength, ylength = image.shape
    vlength = filter_window * filter_window

    ## create 2-D image array and initialize window
    image_array = image.copy()

    ## loop over image with specified window filter_window

    for x in range(window, xlength - (window + 1)):
        for y in range(window, ylength - (window + 1)):
            ## populate window, sort, find median
            filter_window = image[x - window:x + window + 1, y - window:y + window + 1]
            target_vector = filter_window.flatten()
            ## numpy sort
            sorted_array = np.sort(target_vector)
            median = sorted_array[len(target_vector) // 2]
            if threshold <= 0:
                image_array[x, y] = median
            else:
                scale = np.abs(target_vector - median)
                scale = np.sort(scale)
                Sk = 1.4826 * scale[vlength // 2]
                if abs(image_array[x, y] - median) > threshold * Sk:
                    image_array[x, y] = median

    return image_array


def adaptive_mean(image, window=3, threshold=0):
    window = window // 2
    xlength, ylength = image.shape

    image_array = image.copy()

    for x in range(window, xlength):
        for y in range(window, ylength):
            filter_window = image[x - window:x + window + 1, y - window:y + window + 1]
            target_vector = filter_window.flatten()
            mean = filter_window.mean()
            if threshold <= 0:
                image_array[x, y] = mean
            else:
                scale = np.abs(target_vector - mean).mean()
                Sk = 1.4826 * scale
                if abs(image_array[x, y] - mean) > threshold * Sk:
                    image_array[x, y] = mean

    return image_array


def operate_binarization(img, concat=True):
    img2 = img.copy()
    img2 = cv2.GaussianBlur(img2, (0, 0), 1)
    img2 = img2 > np.mean(img2)
    if concat:
        return np.concatenate((img, img2), axis=1)
    else:
        return img2


def operate_square_filter(img, concat=True):
    img2 = cv2.GaussianBlur(img, (0, 0), 1)
    img2 = square_bin_filter(img2, dsize=6, min_square=15)
    if concat:
        return np.concatenate((img, img2), axis=1)
    else:
        return img2


# https://habr.com/ru/post/278435/
def adaptive_binarization_bredly(img, sensitivity=0):
    h, w = img.shape
    integ = integrity_image(img)
    d = h // 16
    ans = np.zeros((h, w), dtype=np.float)
    for i in range(0, h):
        for j in range(0, w):
            x1 = i - d
            x2 = i + d
            y1 = j - d
            y2 = j + d
            if x1 < 0:
                x1 = 0
            if x2 > h:
                x2 = h
            if y1 < 0:
                y1 = 0
            if y2 > w:
                y2 = w
            c = (y2 - y1) * (x2 - x1)
            su = integ[x2, y2] - integ[x1, y2] - integ[x2, y1] + integ[x1, y1]
            if su * (1 - sensitivity) < img[i, j] * c:
                ans[i, j] = 1.0
    return ans


def integrity_image(img):
    h, w = img.shape
    integ = np.zeros((h + 1, w + 1), dtype=np.float)
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            integ[i, j] = integ[i, j - 1] + integ[i - 1, j] - integ[i - 1, j - 1] + img[i - 1, j - 1]
    return integ


def square_bin_filter(img, dsize=6, min_square=20):
    ans = img.copy()
    d = dsize // 2
    h, w = img.shape
    integ = integrity_image(img)
    for i in range(0, h):
        for j in range(0, w):
            x1 = i - d
            x2 = i + d
            y1 = j - d
            y2 = j + d
            if x1 < 0:
                x1 = 0
            if x2 > h:
                x2 = h
            if y1 < 0:
                y1 = 0
            if y2 > w:
                y2 = w
            c = (y2 - y1) * (x2 - x1)
            c = min_square * c / dsize / dsize
            su = integ[x2, y2] - integ[x1, y2] - integ[x2, y1] + integ[x1, y1]
            if su < c:
                ans[i, j] = 0.0
    return ans


def otcy_threshold(img):
    hist, _ = np.histogram(img.ravel(), 256, range=[0, 1])
    cdf = hist.cumsum()
    max_sigma = 0
    max_t = 0
    h, w = img.shape
    n = h * w
    su = np.sum(img)
    cursu = 0
    # determine threshold
    for t in range(1, 255):
        cursu += t / 255 * hist[t]
        m0 = cursu / cdf[t] if cdf[t] > 0 else 0
        w0 = cdf[t] / (h * w)
        m1 = (su - cursu) / (n - cdf[t]) if cdf[t] < n else 0
        w1 = 1 - cdf[t] / (h * w)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = t
    return max_t / 255


# https://habr.com/ru/post/112079/
def adaptive_binarization_otcy(img):
    max_t = otcy_threshold(img)
    return img > max_t


def rgb_to_gray(images):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(images[..., :3], rgb_weights)


def get_sigma(image):
    ans = []
    h, w, *_ = image.shape
    block = 4
    for i in range(500):
        x = random.randint(block, h - block)
        y = random.randint(block, w - block)
        im = image[x - block:x + block, y - block:y + block]
        ans.append(im.std())
    sorted(ans)
    return np.mean(ans[:100])


def adaptive_local_mean_gauss(image, step=8):
    img = image.copy()
    window = step // 2

    h, w, *_ = image.shape
    s = get_sigma(image)
    for x in range(window, h):
        for y in range(window, w):
            filter_window = image[max(x - window, 0):x + window + 1, max(y - window, 0):y + window + 1]
            ml = filter_window.mean()
            sl = filter_window.std()
            if sl != 0:
                img[x, y] = image[x, y] - s / sl * (image[x, y] - ml)
    return img
