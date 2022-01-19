import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from neural_networks.stylegan import *
from tensorflow.keras.applications import ResNet50


def build_resnet50_unet(input_shape):
    """ Input """
    # inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    resnet50.trainable = False
    return resnet50


def build_vgg(input_shape):
    vgg19 = tf.keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    vgg19.trainable = False
    return vgg19


class Encoder:
    def __init__(self, model, steps, encoder=None, n1=None, n2=None):
        self.model = model
        self.steps = steps
        self.loss_hist = []
        self.n1 = n1
        self.n2 = n2
        self.encoder = encoder

    def encode_images(self, gen):
        res1 = []
        res2 = []
        k = len(gen)
        for _ in tqdm(range(k)):
            x1, x2 = self._encode_batch(gen.next())
            res1.append(x1)
            res2.append(x2)
        return np.concatenate(res1), np.concatenate(res2)

    def _encode_batch(self, images):
        n = len(images)
        if self.n1 is None:
            n1 = noise(n)
        else:
            n1 = np.repeat(self.n1[0], n, axis=0)
        if self.n2 is None:
            n2 = nImage(n)
        else:
            n2 = np.repeat(self.n2, n, axis=0)

        x1 = tf.Variable(n1)
        x2 = tf.Variable(n2)
        if self.encoder is None:
            target = tf.constant(images)
        else:
            target = tf.constant(self.encoder(images))

        hist = []
        for lr, nb in tqdm(self.steps):
            for i in range(nb):
                l = self._opt_step(x1, x2, target, lr)
                hist.append(l.numpy())
        self.loss_hist.append(hist)
        return x1.numpy(), x2.numpy()

    def _opt_step(self, x1, x2, target, lr):
        ...


class NoiseEncoder(Encoder):
    def __init__(self, model):
        super().__init__(model, [(0.1, 200), (0.05, 100), (0.01, 100), (0.005, 100)], n1=np.zeros_like(noiseList(1)))

    def _opt_step(self, x1, x2, target, lr):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA(n_layers * [x1] + [x2])
            loss = tf.reduce_sum(tf.abs(result - target), axis=[1, 2, 3])
        dx2 = g.gradient(loss, x2)
        x2.assign(x2 - lr * dx2)
        print(loss)
        return loss

    def transform(self, lattents):
        l = np.repeat(self.n1, len(lattents), axis=1)
        l = list(l) + [lattents]
        return self.model.GAN.GMA(l).numpy()


class StyleEncoder(Encoder):
    def __init__(self, model):
        super().__init__(model, [(0.005, 200), (0.002, 100), (0.001, 100), (0.0005, 100)], n2=np.zeros_like(nImage(1)))

    def _opt_step(self, x1, x2, target, lr):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA([x1] * 7 + [x2])
            loss = tf.reduce_sum(tf.abs(result - target), axis=[1, 2, 3])
        dx1 = g.gradient(loss, x1)
        x1.assign(x1 - lr * dx1)
        print(loss)
        return loss

    def transform(self, lattents):
        n2 = np.repeat(self.n2, len(lattents), axis=0)
        l = np.repeat(np.expand_dims(lattents, axis=0), 7, axis=0)
        l = list(l) + [n2]
        return self.model.GAN.GMA(l).numpy()


class UnionEncoder(Encoder):
    def __init__(self, model):
        super().__init__(model, [(0.002, 200), (0.001, 200), (0.0005, 100), (0.0002, 100), (0.0001, 100)])

    def _opt_step(self, x1, x2, target, lr):
        # opt = tf.keras.optimizers.Adam(learning_rate=lr)
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA([x1] * 7 + [x2])
            loss = tf.reduce_sum(tf.abs(result - target), axis=[1, 2, 3])
            loss = tf.reduce_mean(loss)
            # loss = tf.reduce_mean((result - target) ** 2) * 1e5
        dx1 = g.gradient(loss, x1)
        # opt.apply_gradients(dx1)
        x1.assign(x1 - lr * dx1)
        dx2 = g.gradient(loss, x2)
        # opt.apply_gradients((dx2, x2))
        x2.assign(x2 - lr * dx2)
        print(loss)
        return loss

    def transform(self, x1, x2):
        l = np.repeat(np.expand_dims(x1, axis=0), 7, axis=0)
        l = list(l) + [x2]
        return self.model.GAN.GMA(l).numpy()


class UnionEncoderWithPretrained(Encoder):
    def __init__(self, model, encoder):
        self.encoder = encoder
        super().__init__(model, [(0.2, 200), (0.1, 100), (0.02, 100), (0.01, 100)], encoder=encoder)

    def _opt_step(self, x1, x2, target, lr):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA([x1] * 7 + [x2])
            result = self.encoder(result)
            loss = tf.reduce_sum(tf.abs(result - target), axis=[1, 2, 3])
            loss = tf.reduce_mean(loss)
            # loss = tf.reduce_mean((result - target) ** 2) * 1e5
        dx1 = g.gradient(loss, x1)
        x1.assign(x1 - lr * dx1)
        dx2 = g.gradient(loss, x2)
        x2.assign(x2 - lr * dx2)
        print(loss)
        return loss

    def transform(self, x1, x2):
        l = np.repeat(np.expand_dims(x1, axis=0), 7, axis=0)
        l = list(l) + [x2]
        return self.model.GAN.GMA(l).numpy()


def mixed_style(model, s1, s2, noise):
    tt = int(n_layers / 2)
    p1 = [s1] * tt
    p2 = [s2] * (n_layers - tt)
    latent = p1 + [] + p2
    im = model.GAN.GMA(latent + [noise]).numpy()
    save_images(im, "../results/styleGAN_inverse/style_merge")


if __name__ == "__main__":
    aug = AugmentationUtils() \
        .rescale()

    batch_size = 2
    gen = aug.create_generator("../datasets/test_sem_internet",
                               target_size=(im_size, im_size),
                               batch_size=1,
                               color_mode='rgb',
                               class_mode=None)
    model = StyleGAN(None)
    model.load(19)
    image_dir = "../results/styleGAN_inverse/inv_images"

    # encoder = NoiseEncoder(model)
    # _, nres = encoder.encode_images(gen)
    # save_images(nres, image_dir)
    # images = encoder.transform(nres)
    # save_images(images, image_dir)
    #
    # plt.figure()
    # plt.plot(encoder.loss_hist[0][100:])
    #
    # encoder = StyleEncoder(model)
    # sres, _ = encoder.encode_images(gen)
    # images = encoder.transform(sres)
    # save_images(images, image_dir)
    #
    # plt.figure()
    # plt.plot(encoder.loss_hist[0][100:])
    #
    # l = np.repeat(np.expand_dims(sres, axis=0), 7, axis=0)
    # l = list(l) + [nres]
    #
    # im = model.GAN.GMA(l).numpy()
    # save_images(im, "../results/styleGAN/style_merge")

    encoder = UnionEncoder(model)
    x1, x2 = encoder.encode_images(gen)
    #np.save(output + 'best.npy', best_latent)
    save_images(x2, image_dir)
    images = encoder.transform(x1, x2)
    save_images(images, image_dir)

    plt.figure()
    plt.plot(encoder.loss_hist[0][100:])
    plt.show()
    #
    # mixed_style(model, x1[:1], x1[1:], x2[:1])
    # mixed_style(model, x1[1:], x1[:1], x2[1:])

    nn = x1
    n1 = np.tile(nn, (batch_size, 1))
    n2 = np.repeat(nn, batch_size, axis=0)
    tt = int(n_layers // 2)

    p1 = [n1] * tt
    p2 = [n2] * (n_layers - tt)

    latent = p1 + [] + p2
    res2 = np.repeat(x2, batch_size, axis=0)
    model.generateTruncated(latent, noi=res2, outImage=True, avg=False, trunc=0.0, rim=2)
    model.generateTruncated(latent, noi=res2, outImage=True, avg=True, trunc=0.0, num=1, rim=2)
    model.generateTruncated(latent, noi=res2, outImage=True, avg=True, trunc=0.5, num=2, rim=2)
