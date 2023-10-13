from datetime import datetime
from datetime import timedelta

import tensorflow as tf
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from scipy import linalg
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence

from utils.noisy import *


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.
    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.
    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), name=None, **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None, **kwargs):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        return config


class UnionGenerator(Sequence):

    def __init__(self, generators):

        self.generators = generators
        self.operators = []
        self.batch_size = generators[0].batch_size
        self.samples = generators[0].samples

    def __len__(self):
        return self.generators[0].samples // self.generators[0].batch_size

    def __getitem__(self, idx):
        arrs = []
        for g in self.generators:
            data = g.next()
            if len(data) < self.batch_size:
                data = g.next()
            arrs.append(data)
        for f in self.operators:
            arrs = f(arrs)
        return tuple(arrs)

    def ninty_rotate(self):
        def f(data):
            for i in range(len(data[0])):
                rnd = random.randint(0, 4)
                for j in range(len(data)):
                    data[j][i] = np.rot90(data[j][i], k=rnd)
            return data

        self.operators.append(f)
        return self

    def style_augment(self, model, style_gen):
        def f(data):
            rnd = random.Random()
            pr = rnd.uniform(0.0, 1.0)
            if pr < 0.5:
                images = data[0]
                style = style_gen.next()
                imgs = np.repeat(images, 3, axis=3)
                imgs = model.predict([imgs, style])
                data[0] = tf.reduce_mean(imgs, axis=3, keepdims=True)
            return data

        self.operators.append(f)
        return self

    def next(self):
        return self.__getitem__(-1)


class AugmentationUtils:
    def __init__(self):
        self.params = {}
        self.operators = []

    def create_generator(self, data_dir, target_size=(128, 128), batch_size=16,
                         subset=None, color_mode='grayscale', class_mode=None, seed=None, shuffle=True):
        self.params["preprocessing_function"] = self._augment()
        g = ImageDataGenerator(**self.params)
        return g.flow_from_directory(data_dir,
                                     target_size=target_size,
                                     batch_size=batch_size,
                                     color_mode=color_mode,
                                     class_mode=class_mode,
                                     subset=subset,
                                     shuffle=shuffle,
                                     seed=seed)

    def create_memory_generator(self, data_dir, target_size=(128, 128), batch_size=16,
                                subset=None, gray=True, seed=None):
        self.params["preprocessing_function"] = self._augment()
        read_dir(data_dir, target_size[0], target_size[1], gray=gray)
        g = ImageDataGenerator(**self.params)
        return g.flow(data_dir,
                      batch_size=batch_size,
                      subset=subset,
                      seed=seed)

    def train_val_generator(self, data_dir, target_size=(128, 128), batch_size=16,
                            color_mode='grayscale', class_mode=None, seed=None):
        return self.create_generator(data_dir,
                                     target_size=target_size,
                                     batch_size=batch_size,
                                     color_mode=color_mode,
                                     class_mode=class_mode,
                                     subset='training',
                                     seed=seed), \
            self.create_generator(data_dir,
                                  target_size=target_size,
                                  batch_size=batch_size,
                                  color_mode=color_mode,
                                  class_mode=class_mode,
                                  subset='validation',
                                  seed=seed),

    def _augment(self):
        op = self.operators.copy()

        def augment(img):
            im = img
            for f in op:
                im = f(im)
            return im

        return augment

    def validation_split(self, split=0.2):
        self.params['validation_split'] = split
        return self

    def horizontal_flip(self):
        self.params['horizontal_flip'] = True
        return self

    def vertical_flip(self):
        self.params['vertical_flip'] = True
        return self

    def brightness_range(self, l=0.5, r=1.4):
        self.params['brightness_range'] = (l, r)
        return self

    def zoom_range(self, l=0.7, r=1.1):
        self.params['zoom_range'] = (l, r)
        return self

    def rescale(self, stdNorm=False):
        def f(image):
            image = image / 255.0
            if stdNorm:
                return std_norm_x(image)
            return image

        self.operators.append(f)
        return self

    def std_norm_after_rescale(self):
        def f(image):
            return std_norm_x(image)

        self.operators.append(f)
        return self

    def add_gauss_noise(self, mean=0, var=0.1, p=0.5):
        def f(image):
            return gauss_noise(image, mean, var, p)

        self.operators.append(f)
        return self

    def add_big_light_hole(self, count=3, hl=10, hr=30, wl=10, wr=30, p=0.5):
        def f(image):
            return big_light_hole(image, count, hl, hr, wl, wr, p)

        self.operators.append(f)
        return self

    def add_salt_paper(self, s_vs_p=0.5, amount=0.05, p=0.5):
        def f(image):
            return salt_paper(image, s_vs_p, amount, p)

        self.operators.append(f)
        return self

    def add_light_side(self, coeff=1.5, exponential=False, dist=20, p=0.5):
        def f(image):
            return light_side(image, coeff, exponential, dist, p)

        self.operators.append(f)
        return self

    def add_big_own_defect(self, count=20, hl=5, hr=15, wl=5, wr=15, p=0.5):
        def f(image):
            return big_own_defect(image, count, hl, hr, wl, wr, p)

        self.operators.append(f)
        return self

    def add_defect_expansion_algorithm(self, count=20, sizel=10, sizer=50, gauss=True, p=0.5):
        def f(image):
            return expansion_algorithm(image, count, sizel, sizer, gauss, p)

        self.operators.append(f)
        return self

    def add_unsharp_masking(self, sigma=2.0):
        def f(image):
            return unsharp_masking(image, sigma)

        self.operators.append(f)
        return self

    def add_median_blur(self, k=5, p=1.0):
        def f(image):
            return median_blur(image, k=k, p=p)

        self.operators.append(f)
        return self

    def add_gaussian_blur(self, sigma=1.0, p=1.0):
        def f(image):
            return gaussian_blur(image, sigma=sigma, p=p)

        self.operators.append(f)
        return self

    def ninty_rotation(self):
        def f(image):
            i = random.randint(0, 4)
            return np.rot90(image, k=i)

        self.operators.append(f)
        return self

    def add_different_noise(self, p=0.5):
        def f(image):
            i = random.randint(0, 4)
            if i == 0:
                return gauss_noise(image, var=0.01, p=p)
            if i == 1:
                return salt_paper(image, s_vs_p=0.5, amount=0.05, p=p)
            if i == 2:
                return light_side(image, coeff=64, exponential=False, dist=128, p=p)
            if i == 3:
                return gaussian_blur(image, sigma=2.0, p=p)
            return expansion_algorithm(image, count=20, sizel=10, sizer=50, gauss=True, p=p)

        self.operators.append(f)
        return self

    def random_gauss_noise(self, mean=0, maxvar=1000, p=0.5):
        def f(image):
            var = random.randint(0, maxvar) / maxvar / 10
            return gauss_noise(image, mean, var, p)

        self.operators.append(f)
        return self


def get_gen_images(generator, count=None):
    cnt = 0
    if type(generator) is UnionGenerator:
        images = [[] for _ in range(len(generator.generators))]
        for i in range(len(generator)):
            data = generator.__getitem__(i)
            for i, im in enumerate(data):
                images[i].append(im)
            cnt += data[0].shape[0]
            if count is not None and cnt >= count:
                return [np.concatenate(res)[:count] for res in images]
        return [np.concatenate(res) for res in images]
    else:
        images = []
        for i in range(len(generator)):
            data = generator.__getitem__(i)
            images.append(data)
            cnt += data.shape[0]
            if count is not None and cnt >= count:
                return np.concatenate(images)[:count]
        return np.concatenate(images)


def test_generator(save_dir, generator, count=None, stdNorm=False):
    imgs = get_gen_images(generator, count=count)
    if type(imgs) is list or type(imgs) is tuple:
        imgs = np.hstack(imgs)
    save_images(imgs, save_dir, stdNorm=stdNorm)


def predict_images(gen, images):
    sp = np.expand_dims(images, axis=-1)
    p = gen.predict(sp, batch_size=8)
    return np.reshape(p, p.shape[:-1])


def scan_image(gen, image, img_size, step=8, e=0.01):
    imh, imw, _ = image.shape
    p = []
    ans = np.zeros((imh, imw))
    for i in range(0, imh - img_size + 1, step):
        for j in range(0, imw - img_size + 1, step):
            im = image[i:i + img_size, j:j + img_size]
            p.append(im)
    p = np.asarray(p)
    res = np.zeros((len(p), img_size, img_size))
    start_time = time.time()
    s_pred = 128
    for i in range(0, len(p), s_pred):
        res[i:i + s_pred] = predict_images(gen, p[i:i + s_pred])
    cnt = 0
    summ = np.zeros((imh, imw))
    s = np.ones((img_size, img_size))
    for i in range(0, imh - img_size, step):
        for j in range(0, imw - img_size, step):
            ans[i:i + img_size, j:j + img_size] += res[cnt]
            summ[i:i + img_size, j:j + img_size] += s
            cnt += 1
    summ = summ.astype(np.float)
    ans = np.divide(ans, summ)
    # for i in range(imh):
    #     for j in range(imw):
    #         if abs(ans[i, j] - image[i, j]) <= e:
    #             ans[i, j] = image[i, j]
    print(str(timedelta(seconds=time.time() - start_time)))
    return ans


def test_real_frame(model_path, image_path, model=None, output_path='../results/scan_results/', stdNorm=False,
                    interpolate=False, img_size=128, post_process_func=None, gray=True):
    if model is None:
        model = keras.models.load_model(model_path)
    else:
        model.load_weights(model_path)
    images = read_dir(image_path, 0, 0, gray=gray)
    imageNames = os.listdir(image_path)
    process_real_frame(model, images, output_path=output_path, stdNorm=stdNorm, interpolate=interpolate,
                       imageNames=imageNames, img_size=img_size, post_process_func=post_process_func)


def process_real_frame(gen, images, output_path='../results/scan_results/', stdNorm=False, interpolate=False,
                       imageNames=None, img_size=128, post_process_func=None):
    for i, image in enumerate(images):
        if stdNorm:
            image = std_norm_x(image)
        if not interpolate:
            ar, h, w = split_image(image, img_size, img_size)
            ar = np.asarray(ar)
            p = predict_images(gen, ar)
            im = unionImage(p, h, w)
            res = im
        else:
            res = scan_image(gen, image, img_size, step=img_size // 4)
        if post_process_func is not None:
            res = post_process_func(image, res)
        save_images([res], output_path, stdNorm, imageNames=[imageNames[i]])
        keras.backend.clear_session()

    # estimate quality
    # print(compare_images_by_function(images, res, contrast_measure))
    # print(compare_images_by_function(images, res, sharpness_measure))
    # print(compare_images_by_function(images, res, psnr, True))
    # print(compare_images_by_function(images, res, similarity, True))


def _image_to_image_func(gen, model):
    def f():
        train_x, *_ = gen.next()
        train_x = train_x[:4]
        prediction = model.predict(train_x)
        return train_x, prediction

    return f


class ImageMonitor(keras.callbacks.Callback):

    def __init__(self, train_pred_func):
        self.train_pred_func = train_pred_func

    def on_epoch_end(self, epoch, logs=None):
        fig, ax = plt.subplots(4, 2, figsize=(20, 20))
        fig.suptitle("Epoch " + str(epoch))
        train_x, prediction = self.train_pred_func()
        for i, (img, pred) in enumerate(zip(train_x, prediction)):
            pr = (pred * 255).astype(np.uint8)
            im = (img * 255).astype(np.uint8)
            ax[i, 0].imshow(im, 'gray')
            ax[i, 1].imshow(pr, 'gray')
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")
        plt.show()
        plt.close()


def get_default_callbacks(save_url, gen, model, save_weights_only=True, custom_save=False, monitor_loss='val_loss',
                          mode='auto'):
    return get_callbacks(save_url, _image_to_image_func(gen, model), save_weights_only, custom_save, monitor_loss, mode)


def get_callbacks(save_url, train_pred_func, save_weights_only=True, custom_save=False, monitor_loss='val_loss',
                  mode='auto'):
    now = datetime.now().strftime("%m%d%H:%M")
    if custom_save:
        save_url += '/model' + now
    if not os.path.exists(save_url):
        os.makedirs(save_url)
    monitor = ImageMonitor(train_pred_func)
    if not custom_save:
        save_url += '/model' + now + '.h5'
    mcp_save = ModelCheckpoint(save_url, save_best_only=True,
                               save_weights_only=save_weights_only,
                               verbose=1,
                               monitor=monitor_loss, mode=mode)
    return [monitor, mcp_save]


def create_image_to_image_generator(image_dirs: list, aug_extension=None, stdNorm=False, seed=None,
                                    batch_size=8, im_size=128, color_mode='grayscale', different_seed=False,
                                    vertical_flip=True, ninty_rotate=True, validation_split=0.2):
    if seed is None:
        seed = random.randint(0, 2 ** 30)
    train_gens = []
    val_gens = []
    for i, path in enumerate(image_dirs):
        aug = AugmentationUtils() \
            .rescale(stdNorm) \
            .horizontal_flip() \
            .validation_split(validation_split)
        if vertical_flip:
            aug = aug.vertical_flip()
        if aug_extension is not None and len(aug_extension) > i and aug_extension[i] is not None:
            aug = aug_extension[i](aug)
        if different_seed:
            seed = random.randint(0, 2 ** 30)
        train_generator, val_generator = aug.train_val_generator(path,
                                                                 target_size=(im_size, im_size),
                                                                 batch_size=batch_size,
                                                                 color_mode=color_mode,
                                                                 class_mode=None,
                                                                 seed=seed)
        train_gens.append(train_generator)
        val_gens.append(val_generator)

    train_generator = UnionGenerator(train_gens)
    val_generator = UnionGenerator(val_gens)
    if ninty_rotate:
        train_generator = train_generator.ninty_rotate()
        val_generator = val_generator.ninty_rotate()

    return train_generator, val_generator


def cross_validation(data, targets, k=4):
    num_validation_samples = len(data) // k
    data, targets = shuffle(data, targets)
    for fold in range(k):
        validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        training_data = np.concatenate(
            [data[:num_validation_samples * fold], data[num_validation_samples * (fold + 1):]], axis=0)
        validation_targets = targets[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        training_targets = np.concatenate(
            [targets[:num_validation_samples * fold], targets[num_validation_samples * (fold + 1):]], axis=0)
        yield training_data, training_targets, validation_data, validation_targets


def train_with_cross_val(noisy_data, data_img, recognition_model, k=4):
    k = 4
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    for i in range(20):
        for trainx, trainy, valx, valy in cross_validation(noisy_data, data_img):
            his = recognition_model.fit(trainx, trainy, epochs=1, batch_size=16, validation_data=(valx, valy))
            loss = his.history['loss']
            val_loss = his.history['val_loss']
            acc = his.history['acc']
            val_acc = his.history['val_acc']
            history['loss'].append(loss[0])
            history['val_loss'].append(val_loss[0])
            history['acc'].append(acc[0])
            history['val_acc'].append(val_acc[0])
    plot_graphs(history)
    return recognition_model


def debug_get_activations(model, test_image):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = keras.models.Model(inputs=model.input, outputs=layer_outputs)
    img_tensor = np.expand_dims(test_image, axis=0)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


def get_fid(real_images, fake_images):
    # input_shape = (299, 299, 3)
    model = InceptionV3(include_top=False, pooling='avg', input_shape=real_images[0].shape)
    act1 = model.predict(real_images)
    act2 = model.predict(fake_images)
    return calculate_fid(act1, act2)


# gan loss
def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def recall_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
