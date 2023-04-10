from skimage.metrics import structural_similarity
import cv2
import numpy as np


def mean_psnr(imgs_test, imgs_real):
    res = 0
    for img1, img2 in zip(imgs_test, imgs_real):
        res += psnr(img1, img2)
    return res / len(imgs_test)


def mean_ssim(imgs_test, imgs_real):
    res = 0
    for img1, img2 in zip(imgs_test, imgs_real):
        res += similarity(img1, img2)
    return res / len(imgs_test)


# http://amroamroamro.github.io/mexopencv/matlab/cv.PSNR.html
# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def psnr(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    hm = min(h1, h2)
    wm = min(w1, w2)
    psnr = cv2.PSNR(img1[:hm, :wm], img2[:hm, :wm], 1)
    return psnr


# https://cvnote.ddlee.cc/2019/09/12/psnr-ssim-python
def similarity(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    hm = min(h1, h2)
    wm = min(w1, w2)
    (score, diff) = structural_similarity(img1[:hm, :wm], img2[:hm, :wm], full=True, channel_axis=2)
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
