from time import time
from datetime import timedelta
import keras
from tensorflow.keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from img_filters import *
from noisy import *


class My_Custom_Generator(Sequence):
    def __init__(self, image_path, batch_size, noise_factor):
        self.image_path = image_path
        self.batch_size = batch_size
        self.train_xgenerator = get_image_generator(image_path, batch_size)
        self.cnt = len(self.train_xgenerator)
        self.noise_factor = 1 - noise_factor

    def __len__(self):
        return self.cnt

    def __getitem__(self, idx):
        # batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        # images = np.array([read_image(self.image_path + file_name, 128, 128) for file_name in
        #                  batch_x])
        images, _ = self.train_xgenerator.__next__()
        images = np.reshape(images, images.shape[:-1])
        noisy_data = np.zeros(images.shape)
        d = (1 - self.noise_factor) / 10
        for i, img in enumerate(images):
            p = random.random()
            if p > self.noise_factor:
                # noisy_data[i] = big_light_hole(img)
                # noisy_data[i] = expansion_algorithm(img, 20, gauss=False)
                noisy_data[i] = light_side(img, 5)
                # if p > self.noise_factor + 5 * d:
                #     noisy_data[i] = noisy(img, "gauss")
                # elif p > self.noise_factor + 4 * d:
                #     noisy_data[i] = noisy(img, "s&p")
                # elif p > self.noise_factor + 3 * d:
                #     noisy_data[i] = noisy(img, "light_side")
                # else:
                #     noisy_data[i] = noisy(img, "big_defect")

        # noise_generator = np.random.normal(0, 1, images.shape)
        # save_images(images,"ttt/")
        # save_images(noisy_data, "ttt/")
        # noisy_data = images + self.noise_factor * noise_generator
        noisy_data = np.expand_dims(noisy_data, axis=-1)
        # showImage(images[0])
        images = np.expand_dims(images, axis=-1)
        return noisy_data, images


def get_image_generator(image_path, batch_size):
    g = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, brightness_range=[0.5, 1.4],
                           zoom_range=[0.7, 1.1], rescale=1 / 255.0,
                           preprocessing_function=my_augmented_function)
    return g.flow_from_directory(image_path, target_size=(128, 128),
                                 batch_size=batch_size, color_mode='grayscale')


def save_generator_result(generator):
    for x, val in zip(generator, range(10)):
        pass


def predict_images(gen, images):
    sp = np.expand_dims(images, axis=-1)
    p = gen.predict(sp, batch_size=16)
    return np.reshape(p, p.shape[:-1])


def scan_image(gen, image, h, w):
    imh, imw = image.shape
    # t = [[[] for i in range(imw)] for j in range(imh)]
    p = []
    step = 8
    ans = np.zeros((imh, imw))
    for i in range(0, imh - h, step):
        for j in range(0, imw - w, step):
            im = image[i:i + h, j:j + w]
            p.append(im)
    p = np.asarray(p)
    for i in range(0, len(p), 1024):
        p[i:i + 1024] = predict_images(gen, p[i:i + 1024])
    cnt = 0
    summ = np.zeros((imh, imw))
    s = np.ones((h, w))
    for i in range(0, imh - h, step):
        start_time = time()
        for j in range(0, imw - w, step):
            ans[i:i + h, j:j + w] += p[cnt]
            summ[i:i + h, j:j + w] += s
            cnt += 1
        print(str(timedelta(seconds=time() - start_time)))
    # start_time = time()
    # ans = interpolate_image(imh, imw, p, gl)
    # print(str(timedelta(seconds=time() - start_time)))
    summ = summ.astype(np.float)
    return np.divide(ans, summ)


# @njit(float64[:, :](int64, int64, float32[:, :, :], int64[:, :]), parallel=True)
def interpolate_image(imh, imw, p, gl):
    ans = np.zeros((imh - 16, imw - 16))
    for i in range(0, imh - 16):
        for j in range(0, imw - 16):
            su = 0.0
            k = 0
            for ii, e in enumerate(gl):
                if e[0] <= i < e[1] and e[2] <= j < e[3]:
                    su += p[ii, i - e[0], j - e[2]]
                    k += 1
            # m, _ = stats.mode(np.asarray(ar))
            ans[i, j] = su / k
    return ans


def test_real_frame(model_path, image_path, stdNorm=False, interpolate=False):
    images = read_dir(image_path, 0, 0)
    gen = keras.models.load_model(model_path)
    res = []
    for image in images:
        if stdNorm:
            image = std_norm_x(image)
        if not interpolate:
            ar, h, w = split_image(image, 128)
            ar = np.asarray(ar)
            p = predict_images(gen, ar)
            im = unionImage(p, h, w)
            res.append(im)
        else:
            res.append(scan_image(gen, image, 128, 128))
    save_images(np.asarray(res), 'scan_results/', stdNorm, imageNames=os.listdir(image_path))

    # estimate quality
    # print(compare_images_by_function(images, res, contrast_measure))
    # print(compare_images_by_function(images, res, sharpness_measure))
    # print(compare_images_by_function(images, res, psnr, True))
    # print(compare_images_by_function(images, res, similarity, True))
