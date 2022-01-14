from datetime import timedelta
from tensorflow import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from scipy import linalg
from utils.noisy import *
from tensorflow.keras import layers
from keras.applications.inception_v3 import InceptionV3


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
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, generators, batch_size):

        self.generators = generators
        self.batch_size = batch_size
        self.operators = []

    def __len__(self):
        return self.generators[0].samples // self.batch_size

    def __getitem__(self, idx):
        arrs = []
        for g in self.generators:
            arrs.append(g.next())
        for i in range(len(arrs[0])):
            rnd = random.randint(0, 4)
            for j in range(len(self.generators)):
                for f in self.operators:
                    arrs[j][i] = f(arrs[j][i], rnd)
        return tuple(arrs)

    def reflect_rotate(self):
        def f(img, rnd):
            im = np.rot90(img, k=rnd)
            return im

        self.operators.append(f)
        return self

    def next(self):
        return self.__getitem__(-1)


class AugmentationUtils:
    def __init__(self):
        self.params = {}
        self.operators = []

    def create_generator(self, data_dir, target_size=(128, 128), batch_size=16,
                         subset=None, color_mode='grayscale', class_mode=None, seed=None):
        self.params["preprocessing_function"] = self._augment()
        g = ImageDataGenerator(**self.params)
        return g.flow_from_directory(data_dir,
                                     target_size=target_size,
                                     batch_size=batch_size,
                                     color_mode=color_mode,
                                     class_mode=class_mode,
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
            # im = np.reshape(img, img.shape[:-1])
            im = img
            for f in op:
                im = f(im)
            # return np.expand_dims(im, axis=-1)
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

    def add_gauss_noise(self, mean=0, var=0.1):
        def f(image):
            return gauss_noise(image, mean, var)

        self.operators.append(f)
        return self

    def add_big_light_hole(self, count=3, hl=10, hr=30, wl=10, wr=30):
        def f(image):
            return big_light_hole(image, count, hl, hr, wl, wr)

        self.operators.append(f)
        return self

    def add_salt_paper(self, s_vs_p=0.5, amount=0.05):
        def f(image):
            return salt_paper(image, s_vs_p, amount)

        self.operators.append(f)
        return self

    def add_light_side(self, coeff=1.5, exponential=False):
        def f(image):
            return light_side(image, coeff, exponential)

        self.operators.append(f)
        return self

    def add_big_own_defect(self, count=20, hl=5, hr=15, wl=5, wr=15):
        def f(image):
            return big_own_defect(image, count, hl, hr, wl, wr)

        self.operators.append(f)
        return self

    def add_defect_expansion_algorithm(self, count=20, sizel=10, sizer=50, gauss=True):
        def f(image):
            return expansion_algorithm(image, count, sizel, sizer, gauss)

        self.operators.append(f)
        return self

    def add_unsharp_masking(self, sigma=2.0):
        def f(image):
            return unsharp_masking(image, sigma)

        self.operators.append(f)
        return self

    def add_median_blur(self, k=5):
        def f(image):
            img = np.ascontiguousarray(image, dtype=np.float32)
            cv2.medianBlur(img, k, img)
            return img
        self.operators.append(f)
        return self

    def add_gaussian_blur(self, sigma=1.0):
        def f(image):
            cv2.GaussianBlur(image, (0, 0), sigma, image)
            return image

        self.operators.append(f)
        return self

    def reflect_rotation(self):
        def f(image):
            i = random.randint(0, 4)
            return np.rot90(image, k=i)

        self.operators.append(f)
        return self


def batch_to_image(generator, count=None, wrap=False):
    def wrapper(data, wrap=False):
        if wrap:
            return np.asarray([data])
        else:
            return data

    cnt = 0
    for i in range(len(generator)):
        data = generator.__getitem__(i)
        ok = False
        if type(data) is not list and type(data) is not tuple:
            data = [data]
            ok = True
        for i in range(len(data[0])):
            if not ok:
                yield [wrapper(data[j][i], wrap) for j in range(len(data))]
            else:
                yield [wrapper(data[0][i], wrap)]
            cnt += 1
            if cnt == count:
                return


def get_gen_images(generator, count=1):
    images = [[] for i in range(count)]
    for data in batch_to_image(generator):
        for i, im in enumerate(data):
            images[i].append(im)
    images = [np.asarray(res) for res in images]
    if count > 1:
        return images
    else:
        return images[0]


def test_generator(save_dir, generator, count=None, stdNorm=False):
    imgs = []
    for img in batch_to_image(generator, count=count, wrap=False):
        imgs.append(np.vstack(img))
    save_images(imgs, save_dir, stdNorm=stdNorm)


def predict_images(gen, images):
    sp = np.expand_dims(images, axis=-1)
    p = gen.predict(sp, batch_size=16)
    return np.reshape(p, p.shape[:-1])


def scan_image(gen, image, h, w, step=8, e=0.01):
    imh, imw = image.shape
    # t = [[[] for i in range(imw)] for j in range(imh)]
    p = []
    ans = np.zeros((imh, imw))
    for i in range(0, imh - h, step):
        for j in range(0, imw - w, step):
            im = image[i:i + h, j:j + w]
            p.append(im)
    p = np.asarray(p)
    start_time = time.time()
    for i in range(0, len(p), 1024):
        p[i:i + 1024] = predict_images(gen, p[i:i + 1024])
    cnt = 0
    summ = np.zeros((imh, imw))
    s = np.ones((h, w))
    for i in range(0, imh - h, step):
        for j in range(0, imw - w, step):
            ans[i:i + h, j:j + w] += p[cnt]
            summ[i:i + h, j:j + w] += s
            cnt += 1
    summ = summ.astype(np.float)
    ans = np.divide(ans, summ)
    for i in range(imh):
        for j in range(imw):
            if abs(ans[i, j] - image[i, j]) <= e:
                ans[i, j] = image[i, j]
    print(str(timedelta(seconds=time.time() - start_time)))
    return ans


def test_real_frame(model_path, image_path, output_path='scan_results/', stdNorm=False, interpolate=False):
    gen = keras.models.load_model(model_path)
    return process_real_frame(gen, image_path, output_path=output_path, stdNorm=stdNorm, interpolate=interpolate)


def process_real_frame(gen, image_path, output_path='scan_results/', stdNorm=False, interpolate=False):
    images = read_dir(image_path, 0, 0)
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
    save_images(np.asarray(res), output_path, stdNorm, imageNames=os.listdir(image_path))

    # estimate quality
    # print(compare_images_by_function(images, res, contrast_measure))
    # print(compare_images_by_function(images, res, sharpness_measure))
    # print(compare_images_by_function(images, res, psnr, True))
    # print(compare_images_by_function(images, res, similarity, True))
    return np.asarray(res)


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
