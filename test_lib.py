from scipy.stats import norm
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from neural_networks.unet import *
from utils import canny_edge_detector as canny
from utils.regression import *
import utils.mylibrary as lib

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
        showImage(noisy_with_defaults(test_img[i], "gauss"))
        showImage(noisy_with_defaults(test_img[i], "s&p"))
        showImage(noisy_with_defaults(test_img[i], "big_defect"))
        showImage(noisy_with_defaults(test_img[i], "light_side"))
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
    recursive_read_operate_save('/home/nik/images/', '/home/nik/test_square_filter2/',
                                operate_binarization)


def morph_repair():
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


def experiments():
    fn = 'train_images/img302.png'
    img = read_image(fn, 128, 128)
    showImage(img)
    # img = unsharp_masking(img, 3)
    img = adaptive_median(img, threshold=1.0)
    # showImage(img)
    # ans = operate_binarization(img)
    # showImage(ans)
    ans = square_bin_filter(img, dsize=6, min_square=15)
    showImage(ans)


# https://newbedev.com/how-to-use-ridge-detection-filter-in-opencv
def detect_ridges(gray, sigma=1.0):
    H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
    maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
    return maxima_ridges, minima_ridges


def plot_images(*images):
    images = list(images)
    n = len(images)
    fig, ax = plt.subplots(ncols=n, sharey=True)
    for i, img in enumerate(images):
        ax[i].imshow(img, cmap='gray')
        ax[i].axis('off')
    plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
    plt.show()


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
    plot_images(img, a, b)


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
    lib.inn = 0
    prepare_dataset("datasets/unet/want_to_split/clear_masks", 'datasets/unet/all_mask_images/train/', 256, step=256,
                    drop=0,
                    determined=True)
    train_generator, val_generator = create_image_to_image_dataset(
        ['datasets/unet/all_real_images', 'datasets/unet/all_mask_images'], im_size=256)
    test_generator("results/test", train_generator, 200)
    test_generator("results/test", val_generator, 200)


if __name__ == '__main__':
    # recursive_read_operate_save('train_images/', 'all_bin_images/', image_complex_bin, False)
    # copy_from_labels("unet/bin_images", "train_images", "unet/real_images")
    # test_real_frame('models/rednet/model020122:19.h5', "scan_images/", stdNorm=False, interpolate=True)
    # recursive_read_operate_save('datasets/one_layer_images/one_cadr', 'results/binary', simple_threshold, False)

    # recursive_read_operate_save('datasets/cycle/sem_to_sem/imgsA/train', 'datasets/one_layer_images/splited1/train', simple_threshold, False)

    # prepare_dataset('datasets/one_layer_images/want_to_split_bin', 'datasets/one_layer_images/splited1/train', 256, step=256, drop=0)
    train_generator, val_generator = create_image_to_image_dataset(
        ['datasets/unet/all_real_images', 'datasets/unet/all_mask_images'], im_size=256)
    test_generator("results/test", train_generator, 200)
    test_generator("results/test", val_generator, 200)
    pass
