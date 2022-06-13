import math
import random

import numpy as np
from scipy.signal import convolve2d
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from tqdm import tqdm

from neural_networks.adain import NeuralStyleTransfer
from utils import canny_edge_detector as canny
from utils.mykeras_utils import *
from utils.noisy import noisy_with_defaults, expansion_algorithm, light_side
from utils.regression import *
from utils.img_filters import get_sigma
import timeit


def test_filters():
    width = 128
    height = 128
    test_img = read_dir("Simple_dataset/Good", height, width, True)
    for i in range(len(test_img)):
        test_img[i] = custom_algorithm(test_img[i])
    save_images(test_img, "results/contrast_dataset/", stdNorm=False)


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
        showImage(noisy_with_defaults(test_img[i], "gauss"))
        showImage(noisy_with_defaults(test_img[i], "s&p"))
        showImage(noisy_with_defaults(test_img[i], "big_defect"))
        showImage(noisy_with_defaults(test_img[i], "light_side"))
        showImage(test_img[i])
    save_images(test_img, "results/noisy_dataset/", stdNorm=False)


def final_generation():
    im = np.ones((128, 128, 1), dtype=np.float64) * 0.2
    im = big_own_defect(im, 10)
    # im = expansion_algorithm(im, 10, sizel=60, sizer=120, gauss=False)
    showImage(im)
    save_images([im], "results/article", imageNames=["big_own.png"])


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
    img = read_image("datasets/test/dog.jpg", 160, 120)
    showImage(img)
    img = light_side(img, 10)
    showImage(img)
    img = adaptive_hist(img, 20)
    showImage(img)
    cv2.imwrite('results/article/light.jpg', 255 * img)


def collision():
    fn = 'train_images/img13.png'
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
    cv2.imwrite('results/article/collision.jpg', 255 * img)
    cv2.imwrite('results/article/collisionsrc.jpg', 255 * src)


def median_std():
    fn = 'train_images/img302.png'
    img = read_image(fn, 128, 128)
    img = expansion_algorithm(img, 20, gauss=False)
    showImage(img)
    img = img * 255
    img = img.astype(np.uint8)
    final = cv2.medianBlur(img, 7)
    showImage(final / 255)
    cv2.imwrite('results/article/median.jpg', final)
    cv2.imwrite('results/article/mediansrc.jpg', img)


def adaptive():
    img = read_image("datasets/test/cat.jpg", 120, 160)
    img = expansion_algorithm(img, 20, gauss=False)
    showImage(img)
    # print(timeit.timeit('adaptive_median(img, threshold=1.0)', number=10,
    #                     globals={'adaptive_median': adaptive_median, 'img': img, }) / 10)
    img = adaptive_median(img, threshold=0.1)
    showImage(img)
    cv2.imwrite('results/article/admedian.jpg', 255 * img)


def morph_repair():
    img = read_image("datasets/test/cat.jpg", 120, 160)
    img = expansion_algorithm(img, 20, gauss=True)
    showImage(img)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    res = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    showImage(res)
    cv2.imwrite("results/article/morph.jpg", 255 * res)


# https://newbedev.com/how-to-use-ridge-detection-filter-in-opencv
def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


def canny_with_morph():
    fn = 'datasets/one_layer_images/one_cadr/1129 (1).jpg'
    img = read_image(fn)
    showImage(img)
    ans = canny.detect(img * 255)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ans = cv2.morphologyEx(ans / 255, cv2.MORPH_CLOSE, se)
    showImage(ans)
    # ans = hessian(255*img, mode='constant')
    # showImage(ans / 255)
    a, b = detect_ridges(img, sigma=3.0)
    showImage(np.concatenate([img, a, b], axis=1))


def ridge_detector_test():
    fn = 'one_layer_images/train_images/img302.png'
    img = read_image(fn, 128, 128)
    # img = adaptive_median(img, threshold=1.0)
    showImage(img)
    ans = ridge_detector(img)
    showImage(ans)
    # cv2.imwrite("ridgedetector.png", ans * 255)


def image_complex_bin(img):
    sq = operate_square_filter(img, False)
    cv2.imwrite("results/binary/sq.png", sq * 255)
    dsq = operate_square_filter(sq, False)
    cv2.imwrite("results/binary/dsq.png", dsq * 255)
    bin = operate_binarization(img, False)
    cv2.imwrite("results/binary/bin.png", bin * 255)
    ans = bin + sq
    cv2.imwrite("results/binary/ans.png", ans * 255)

    ans = (ans > 0).astype(np.float)
    cv2.imwrite("results/binary/meanans.png", ans * 255)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ans = cv2.morphologyEx(ans, cv2.MORPH_OPEN, se)
    cv2.imwrite("results/binary/morphans.png", ans * 255)
    cv2.imwrite("results/binary/concat.png", ans * img * 255)
    return ans


def get_dataset():
    # prepare_dataset('/home/nik/images/', 'datasets/global_images/', 256, step=256, drop=0.9)
    prepare_dataset('datasets/balanced_images', 'datasets/final_good_images/train/', 256, step=256, drop=0)


def get_dataset2():
    # prepare_dataset('/home/nik/images/', 'datasets/global_images/', 256, step=256, drop=0.9)
    prepare_dataset('datasets/one_layer_images/want_to_split', 'datasets/cycle/sem_to_sem256/imgsA/train', 256,
                    step=64, drop=0.5)
    prepare_dataset('datasets/one_layer_images/want_to_split2', 'datasets/cycle/sem_to_sem256/imgsB/train', 256,
                    step=64, drop=0.5)


def adaptive1_bad():
    fn = 'datasets/one_layer_images/one_cadr/1129 (1).jpg'
    img = read_image(fn)
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (0, 0), 1)
    showImage(img)
    # img = unsharp_masking(img, 3)
    # img = adaptive_median(img, threshold=1.0)
    # showImage(img)
    # ans = adaptive_binarization_bredly(img)
    ans = adaptive_binarization_otcy(img)
    # showImage(ans)
    # ans = square_bin_filter(img, dsize=6, min_square=20)
    im2 = np.clip(img + ans, 0.0, 1.0)
    sx = sobel_x(im2)
    cv2.imwrite("results/binary/sobel.png", sx * 255)
    sx = np.logical_and(0.1 < sx, sx < 0.15)
    sx = cv2.medianBlur(sx.astype(np.float32), 3)
    sx = cv2.medianBlur(sx, 3)
    cv2.imwrite("results/binary/sobel2.png", sx * 255)
    cv2.imwrite("results/binary/sobel3.png", np.clip(sx * 0.5 + ans, 0.0, 1.0) * 255)


def brute_threshold():
    fn = 'datasets/one_layer_images/one_cadr/cy2_m1_0435.jpg'
    img = read_image(fn)
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (0, 0), 1)
    images = []
    for i in range(0, 256, 1):
        images.append(img >= i / 256)
    save_images(np.asarray(images), "results/binary/brute_threshold")


def simple_threshold(img):
    img = cv2.medianBlur(img, 5)
    img = cv2.GaussianBlur(img, (0, 0), 1)
    ans = img >= 90 / 256
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ans = cv2.morphologyEx(ans.astype(np.float32), cv2.MORPH_OPEN, se)
    return ans


def check_sem():
    dir_name1 = "datasets/unet/want_to_split/real"
    dir1 = sorted(os.listdir(dir_name1))[200:]
    imgs1 = []
    for l in dir1:
        im = read_image(os.path.join(dir_name1, l))
        if im.shape[0] == 4000:
            imgs, _, _ = split_image(im, 2000, 2000)
            imgs1.extend(imgs)
        else:
            imgs1.append(im)
    dir_name2 = "datasets/unet/want_to_split/clear_masks"
    dir2 = sorted(os.listdir(dir_name2))[200:]
    imgs2 = []
    for l in dir2:
        im = read_image(os.path.join(dir_name2, l))
        if im.shape[0] == 4000:
            imgs, _, _ = split_image(im, 2000, 2000)
            imgs2.extend(imgs)
        else:
            imgs2.append(im)
    # dir2 = [st.split('[')[0][:-6] + st.split('[')[-1][-4:] for st in dir2]
    # save_images(imgs2, "datasets/unet/want_to_split/clear_masks", imageNames=dir2)
    for i in range(len(imgs1)):
        imgs1[i] = np.concatenate([imgs1[i], imgs2[i]])
        # print(dir1[i][:-4], dir2[i][:-4])
    save_images(imgs1, "results/test", )


def create_unet_dataset():
    prepare_dataset("datasets/unet/want_to_split/real", 'datasets/unet/all_real_images/train/', 256, step=256, drop=0,
                    determined=True)
    save_images.inn = 0
    prepare_dataset("datasets/unet/want_to_split/clear_masks", 'datasets/unet/all_mask_images/train/', 256, step=256,
                    drop=0,
                    determined=True)
    train_generator, val_generator = create_image_to_image_generator(
        ['datasets/unet/all_real_images', 'datasets/unet/all_mask_images'], im_size=256)
    test_generator("results/test", train_generator, 200)
    test_generator("results/test", val_generator, 200)


def test_unet():
    from neural_networks.unet import get_unet_model

    test_real_frame(get_latest_filename('models/unet/'), 'datasets/unet/test', output_path='results/unet/scan_results',
                    model=get_unet_model((256, 256)), img_size=256,
                    interpolate=True, post_process_func=lambda x, y: simple_boundary(y))


def test_adain():
    img_size = (256, 256)
    style_model = NeuralStyleTransfer((*img_size, 3), alpha=0.01)
    style_model.load_weights(get_latest_filename("models/adain/"))
    aug = AugmentationUtils().rescale()
    content = aug.create_generator('datasets/final_good_images', target_size=img_size,
                                   color_mode='rgb')
    style = aug.create_generator('datasets/style', target_size=img_size,
                                 color_mode='rgb')
    content = get_gen_images(content, 32)
    style = get_gen_images(style, 64)
    for i in tqdm(range(len(content))):
        imgs = np.repeat(content[i:i + 1], 64, axis=0)
        for j in range(0, 64, 8):
            style_imgs = style_model.predict([imgs[j:j + 8], style[j:j + 8]])
            save_images(np.concatenate([imgs[j:j + 8], style[j:j + 8], style_imgs], axis=2), "results/style/adain")


def oct_prepare():
    features = read_dir("datasets/not_sem/OCT_dataset_raw/images", 256, 256, sort=True)
    labels = read_dir("datasets/not_sem/OCT_dataset_raw/labels", 256, 256, sort=True)
    final_features = []
    final_labels = []
    for i in range(len(features)):
        if np.mean(labels[i]) < 0.5:
            final_labels.append(labels[i])
            final_features.append(features[i])
        else:
            print("large")
    save_images(final_features, "datasets/not_sem/OCT_dataset/real/images")
    save_images.inn = 0
    save_images(final_labels, "datasets/not_sem/OCT_dataset/masks/images")


def classical_pipeline(img):
    img = img.astype(np.float32)
    img = cv2.medianBlur(img, 5)  # adaptive_median(img, threshold=1.0)
    # img = cv2.GaussianBlur(img, (0, 0), 1)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ans = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    return ans


def calculate_metrics():
    from neural_networks.encoder_decoder import create_enc_dec, noise_function
    from utils.image_quality import compare_images_by_function, psnr, similarity, contrast_measure
    repair_model = create_enc_dec()
    repair_model.load_weights(get_latest_filename("models/enc_cats"))
    train_generator, val_generator = create_image_to_image_generator(
        ['datasets/not_sem/cats/real', 'datasets/not_sem/cats/real'],
        aug_extension=[noise_function],
        batch_size=1,
        im_size=256, vertical_flip=False,
        ninty_rotate=False)
    total = 500
    imagex = []
    imageny = []
    imagecy = []
    imageay = []
    imagewy = []
    imagey = []
    for i in tqdm(range(total)):
        x, y = train_generator.next()
        ny = repair_model.predict(x)
        imagey.append(np.squeeze(y))
        imagex.append(np.squeeze(x))
        imageny.append(np.squeeze(ny))
        x = np.squeeze(x)
        imagecy.append(classical_pipeline(x))
        imageay.append(adaptive_local_mean_gauss(x))
        imagewy.append(standart_winner(x))
    print(compare_images_by_function(imagex, imagey, psnr, double=True))
    print(compare_images_by_function(imagex, imageny, psnr, double=True))
    print(compare_images_by_function(imagex, imagecy, psnr, double=True))
    print(compare_images_by_function(imagex, imageay, psnr, double=True))
    print(compare_images_by_function(imagex, imagewy, psnr, double=True))
    print(compare_images_by_function(imagex, imagey, similarity, double=True))
    print(compare_images_by_function(imagex, imageny, similarity, double=True))
    print(compare_images_by_function(imagex, imagecy, similarity, double=True))
    print(compare_images_by_function(imagex, imageay, similarity, double=True))
    print(compare_images_by_function(imagex, imagewy, similarity, double=True))
    print(compare_images_by_function(imagex, imagey, contrast_measure))
    print(compare_images_by_function(imageny, imagecy, contrast_measure))
    print(compare_images_by_function(imageay, imagewy, contrast_measure))
    imagex = np.asarray(imagex)
    imagecy = np.asarray(imagecy)
    save_images(np.concatenate([imagex, imagecy], axis=2), "results/test")


def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(kernel, s=img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(np.fft.ifft2(dummy))
    return dummy.astype(np.float32)


def standart_winner(img):
    kernel_size = 3
    kernel = np.eye(kernel_size) / kernel_size
    return wiener_filter(img, kernel, 0.05)


def gaussian_kernel(size, sigma=1.0):
    gausskernel = np.zeros((size, size), np.float32)
    for i in range(size):
        for j in range(size):
            norm = math.pow(i - 1, 2) + pow(j - 1, 2)
            gausskernel[i, j] = math.exp(-norm / (2 * math.pow(sigma, 2)))
    sum = np.sum(gausskernel)
    kernel = gausskernel / sum
    return kernel


def blur(img, kernel_size=3):
    dummy = np.copy(img)
    h = np.eye(kernel_size) / kernel_size
    dummy = convolve2d(dummy, h, mode='valid')
    return dummy


def wiener_test():
    img = read_image("datasets/test/dog.jpg", 120, 160) * 255

    img = img[:120, :120]
    # Blur the image
    kernel_size = 3
    blurred_img = blur(img, kernel_size=kernel_size)

    # Add Gaussian noise

    noisy_img = gauss_noise(blurred_img, var=36)

    # Apply Wiener Filter
    # kernel = gaussian_kernel(5, sigma=6)
    # noisy_img = img
    kernel = np.eye(kernel_size) / kernel_size

    filtered_img = wiener_filter(noisy_img, kernel, K=0.05)
    showImage(img / 255)
    showImage(blurred_img / 255)
    showImage(noisy_img / 255)
    showImage(filtered_img / 255)


def define_noise():
    img = read_image("datasets/test/dog.jpg", 120, 160)
    img = gauss_noise(img, var=0.016)
    ss = 0.016 ** 0.5
    showImage(img)
    s = get_sigma(img)
    print(s)


def compare_gauss():
    from dyplom_experements import REDnet
    from sklearn.metrics import mean_squared_error
    im_size = 256
    autoencoder = REDnet.REDNet(num_layers=10, channels=1)
    autoencoder.build((None, im_size, im_size, 1))
    autoencoder.load_weights(get_latest_filename("models/rednetcats"))
    ss = np.linspace(0, 1.0, 20)
    aug = AugmentationUtils().rescale()
    content = aug.create_generator('datasets/not_sem/cats/real', target_size=[im_size, im_size],
                                   color_mode='grayscale')
    images = get_gen_images(content, 10)
    classic = []
    neural = []
    for s in tqdm(ss):
        cm = 0
        nm = 0
        for im in images:
            img = gauss_noise(im, var=s * s)
            res = adaptive_local_mean_gauss(img)
            cm += mean_squared_error(res[..., 0], im[..., 0])
            img = np.expand_dims(img, axis=0)
            res = autoencoder.predict(img)[0]
            nm += mean_squared_error(res[..., 0], im[..., 0])
        classic.append(cm / 10)
        neural.append(nm / 10)

    fig, ax = plt.subplots()
    ax.plot(ss, classic, label='adaptive mean')
    ax.plot(ss, neural, label='neural')
    # ax.title('Graph')
    ax.legend()
    plt.xlabel('sigma')
    plt.ylabel('mse')
    plt.show()


def benchmark_algorithms():
    from dyplom_experements import REDnet
    im_size = 256
    img = read_image("datasets/test/dog.jpg", im_size, im_size)
    img = gauss_noise(img, var=0.1)
    autoencoder = REDnet.REDNet(num_layers=10, channels=1)
    autoencoder.build((None, im_size, im_size, 1))
    autoencoder.load_weights(get_latest_filename("models/rednetcats"))
    img_n = np.expand_dims(img, axis=-1)
    img_n = np.expand_dims(img_n, axis=0)
    number = 20
    print('autoencoder.predict(img)', timeit.timeit('autoencoder.predict(img)', number=number,
                                                    globals={'autoencoder': autoencoder, 'img': img_n, }))

    print('classical_pipeline(img)', timeit.timeit('classical_pipeline(img)', number=number,
                                                   globals={'classical_pipeline': classical_pipeline, 'img': img, }))

    print('adaptive_local_mean_gauss(img)', timeit.timeit('adaptive_local_mean_gauss(img)', number=number,
                                                          globals={
                                                              'adaptive_local_mean_gauss': adaptive_local_mean_gauss,
                                                              'img': img, }))
    kernel_size = 3
    kernel = np.eye(kernel_size) / kernel_size

    print('wiener_filter(img, kernel, K=0.05)', timeit.timeit('wiener_filter(img, kernel, K=0.05)', number=number,
                                                              globals={'wiener_filter': wiener_filter, 'kernel': kernel,
                                                                       'img': img, }))


if __name__ == '__main__':
    # recursive_read_operate_save('train_images/', 'all_bin_images/', image_complex_bin, False)
    # copy_from_labels("unet/bin_images", "train_images", "unet/real_images")
    # test_real_frame('models/rednet/model020122:19.h5', "scan_images/", stdNorm=False, interpolate=True)
    # recursive_read_operate_save('datasets/one_layer_images/one_cadr', 'results/binary', simple_threshold, False)

    # recursive_read_operate_save('datasets/cycle/sem_to_sem/imgsA/train', 'datasets/one_layer_images/splited1/train', simple_threshold, False)

    # prepare_dataset('datasets/one_layer_images/want_to_split_bin', 'datasets/one_layer_images/splited1/train', 256, step=256, drop=0)

    #
    # recursive_read_operate_save('datasets/not_sem/cats/annotations', 'datasets/not_sem/cats/an2',
    #                             lambda x: x * 255 != 2.0, False)

    # img = read_image("results/article/img291.png")
    # im1 = img[:, :256]
    # cv2.imwrite("results/article/img1.png", np.rot90(im1, 3) * 255)
    # im1 = img[:, 256:512]
    # cv2.imwrite("results/article/img2.png", np.rot90(im1, 3) * 255)
    # im1 = img[:, 512:]
    # cv2.imwrite("results/article/img3.png", np.rot90(im1, 3) * 255)

    # wiener_test()
    # adaptive()
    # morph_repair()
    # light()
    # define_noise()
    # compare_gauss()
    benchmark_algorithms()
    #calculate_metrics()
    pass
