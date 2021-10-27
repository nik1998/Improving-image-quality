import os

from mylibrary import *
from img_filters import *
from mykeras_utils import *
from noisy import *
from scipy.stats import truncnorm, norm
import keras
import cv2


def test_filters():
    width = 128
    height = 128
    test_img = read_dir("Simple_dataset/Good", height, width, True)
    for i in range(len(test_img)):
        test_img[i] = custom_algorithm(test_img[i])
    save_images(test_img, "Contrast_dataset/", stdNorm=False)


def enc_dec_test():
    height, width = 128, 128
    test = read_dir("test_images", height, width)
    test = np.expand_dims(test, axis=-1)
    en_dec_model = keras.models.load_model('models/model08/07/2021, 10:29:56.h5')
    bsize = 16
    p = en_dec_model.predict(test, batch_size=bsize)
    test = np.reshape(test, test.shape[:-1])
    p = np.reshape(p, p.shape[:-1])
    unsharp_masking(p)
    unsharp_masking(p)
    unionTestImages(test, p, path="unionTest2/", stdNorm=False)


def noisy_test():
    height, width = 128, 128
    test_img = read_dir("Simple_dataset/Good", height, width, True)
    for i in range(len(test_img)):
        showImage(test_img[i])
        showImage(noisy(test_img[i], "gauss"))
        showImage(noisy(test_img[i], "s&p"))
        showImage(noisy(test_img[i], "big_defect"))
        showImage(noisy(test_img[i], "light_side"))
        showImage(test_img[i])
    save_images(test_img, "noisy_dataset/", stdNorm=False)


def final_interpolation():
    # test_real_frame('models/gan/generator.h5', "scan_images/",stdNorm=True)
    # test_real_frame('models/model09/09/enc_dec.h5', "scan_images/", stdNorm=False, interpolate=True)
    print(norm.cdf(3, 0, 1))

    im = np.ones((128, 128), dtype=np.float) * 0.2
    # im = big_own_defect(im, 20)
    im = expansion_algorithm(im, 20)
    showImage(im)


# semiconductor manufacturing SEM images
def double_bounds():
    print(__doc__)

    fn = 'train_images/img121.png'
    img = read_image(fn, 128, 128)
    src = img.copy()
    showImage(img)
    img = total_masking(img, 1, 3)
    showImage(img)
    # img = adaptive_hist(img, 20)
    # showImage(img)
    img = (255 * (img > 0.5)).astype(np.uint8)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    im1 = img.copy()
    cv2.drawContours(im1, contours, -1, (255, 0, 0), 4, cv2.LINE_AA, hierarchy, 1)
    cv2.imshow('contours1', im1)
    im2 = img.copy()
    cv2.drawContours(im2, contours, -1, (255, 0, 0), 2, cv2.LINE_AA, hierarchy, 1)
    bounds = im1 - im2
    cv2.imshow('contours2', bounds)
    # src = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)
    # cv2.imshow('contours2', src)
    bounds = cv2.GaussianBlur(bounds, (0, 0), 1)
    res = src + bounds.astype(np.float) / 255 / 2
    showImage(res)
    res = res * 255
    cv2.imwrite('result/im1.jpg', res)

    cv2.waitKey()
    cv2.destroyAllWindows()


def light():
    fn = 'cat_dogs/1.jpg'
    img = read_image(fn, 128, 128)
    src = img.copy()
    showImage(img)
    img = light_side(img, 10)
    showImage(img)

    cv2.imwrite('result/light1.jpg', 255 * img)
    cv2.imwrite('result/light1src.jpg', 255 * src)


def collision():
    # recursive_read_split("semiconductor/", 128, False, 0.5)
    fn = 'train_images2/img13.png'
    img = read_image(fn, 128, 128)
    src = img.copy()
    showImage(img)
    img = cv2.GaussianBlur(img, (0, 0), 1)
    img = img > np.mean(img)
    img = img * np.mean(img) / 4
    img = cv2.GaussianBlur(img, (0, 0), 2)
    img = np.transpose(img)
    showImage(img + src)
    img = img + src
    cv2.imwrite('result/collision.jpg', 255 * img)
    cv2.imwrite('result/collisionsrc.jpg', 255 * src)


def median_std():
    fn = 'train_images/img302.png'
    img = read_image(fn, 128, 128)
    img = expansion_algorithm(img, 20, gauss=False)
    showImage(img)
    img = img * 255
    img = img.astype(np.uint8)
    final = cv2.medianBlur(img, 7)
    showImage(final / 255)
    cv2.imwrite('result/median.jpg', final)
    cv2.imwrite('result/mediansrc.jpg', img)


def adaptive():
    fn = 'train_images/img302.png'
    img = read_image(fn, 128, 128)
    img = expansion_algorithm(img, 20, gauss=False)
    showImage(img)
    img = adaptive_median(img, threshold=1.0)
    showImage(img)
    cv2.imwrite('result/admedian.jpg', 255 * img)


def global_test():
    recursive_read_operate_save('/home/nik/images/', '/home/nik/test_images/',
                                operate_binarization)


if __name__ == '__main__':
    fn = 'improve/4.103 check_0477.jpg'
    img = read_image(fn)
    img = img[-150:-22, -150:-202]
    showImage(img)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    showImage(res)
    cv2.imwrite("improve_results/morphfinal.png", 255 * np.concatenate((img, res), axis=1))
    cv2.imwrite("improve_results/morphfinal2.png",
                255 * np.concatenate((operate_binarization(img), operate_binarization(res)), axis=1))
