import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from tqdm import tqdm

from neural_networks.stylegan import *


class NoiseEncoder:
    def __init__(self, model):
        self.model = model
        self.steps = [(0.1, 200), (0.05, 100), (0.01, 100), (0.005, 100)]
        self.loss_hist = []
        self.n1 = noiseList(1)

    def _opt_step(self, x1, x2, target, lr):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA(n_layers * [x1] + [x2])
            loss = tf.reduce_sum(tf.abs(result - target))
        dx2 = g.gradient(loss, x2)
        x2.assign(x2 - lr * dx2)
        return loss

    def _encode_batch(self, images):
        n = len(images)
        n1 = np.repeat(self.n1[0], n, axis=0)
        x1 = tf.Variable(n1)
        n2 = nImage(n)
        x2 = tf.Variable(n2)
        target = tf.constant(images)

        hist = []
        for lr, nb in self.steps:
            for i in range(nb):
                l = self._opt_step(x1, x2, target, lr)
                hist.append(l.numpy())
        self.loss_hist.append(hist)

        return x2.numpy()

    def encode_images(self, gen):
        res = []
        k = len(gen)
        for _ in tqdm(range(k)):
            res.append(self._encode_batch(gen.next()))
        return np.concatenate(res)

    def transform(self, lattents):
        images = []
        for lattent in lattents:
            n2 = tf.expand_dims(lattent, 0)
            images.append(self.model.GAN.GMA(self.n1 + [n2])[0].numpy())
        return images


class StyleEncoder:
    def __init__(self, model):
        self.model = model
        self.steps = [(0.1, 200), (0.05, 100), (0.01, 100), (0.005, 100)]
        self.loss_hist = []
        self.n2 = nImage(1)

    def _opt_step(self, x1, x2, target, lr):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA(tf.unstack(x1) + [x2])
            loss = tf.reduce_sum(tf.abs(result - target))
        dx1 = g.gradient(loss, x1)
        x1.assign(x1 - lr * dx1)
        return loss

    def _encode_batch(self, images):
        n = len(images)
        n1 = noiseList(n)
        x1 = tf.Variable(n1)
        n2 = np.repeat(self.n2, n, axis=0)
        x2 = tf.Variable(n2)
        target = tf.constant(images)

        hist = []
        for lr, nb in self.steps:
            for i in range(nb):
                l = self._opt_step(x1, x2, target, lr)
                hist.append(l.numpy())
        self.loss_hist.append(hist)

        return x1.numpy(), x2.numpy()

    def encode_images(self, gen):
        res = []
        k = len(gen)
        for _ in tqdm(range(k)):
            res.append(self._encode_batch(gen.next()))
        return np.concatenate(res)

    def transform(self, lattents):
        n2 = np.repeat(self.n2, len(lattents[0]), axis=0)
        l = list(lattents) + [n2]
        return self.model.GAN.GMA(l).numpy()


class UnionEncoder:
    def __init__(self, model):
        self.model = model
        self.steps = [(0.1, 200), (0.05, 100), (0.01, 100), (0.005, 100)]
        self.loss_hist = []

    def _opt_step(self, x1, x2, target, lr):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA(tf.unstack(x1) + [x2])
            #loss = tf.reduce_mean(tf.abs(result - target))
            loss = tf.reduce_mean((result - target) ** 2)
        dx1 = g.gradient(loss, x1)
        x1.assign(x1 - lr * dx1)
        dx2 = g.gradient(loss, x2)
        x2.assign(x2 - lr * dx2)
        # print(x2.numpy())
        print(loss)
        print(lr)
        return loss

    def _encode_batch(self, images):
        n = len(images)
        n1 = noiseList(n)
        x1 = tf.Variable(n1)
        n2 = nImage(n)
        x2 = tf.Variable(n2)
        target = tf.constant(images)

        hist = []
        for lr, nb in self.steps:
            for i in range(nb):
                l = self._opt_step(x1, x2, target, lr)
                hist.append(l.numpy())
        self.loss_hist.append(hist)

        return x1.numpy(), x2.numpy()

    def encode_images(self, gen):
        res1 = []
        res2 = []
        k = len(gen)
        for _ in tqdm(range(k)):
            x1, x2 = self._encode_batch(gen.next())
            res1.append(x1)
            res2.append(x2)
        return np.concatenate(res1), np.concatenate(res2)

    def transform(self, x1, x2):
        l = list(x1) + [x2]
        return self.model.GAN.GMA(l).numpy()


if __name__ == "__main__":
    aug = AugmentationUtils() \
        .rescale()

    gen = aug.create_generator("../datasets/test_sem_internet",
                               target_size=(im_size, im_size),
                               batch_size=BATCH_SIZE,
                               color_mode='rgb',
                               class_mode=None)
    model = StyleGAN(gen, lr=0.0001, silent=False)
    # model.save(100)
    model.load(20)

    # encoder = NoiseEncoder(model)
    # res = encoder.encode_images(gen)
    # images = encoder.transform(res)
    # save_images(images, "../results/styleGAN/inverce_im")

    # plt.figure()
    # plt.plot(encoder.loss_hist[0])

    # nn = noise(8)
    # n1 = np.tile(nn, (8, 1))
    # n2 = np.repeat(nn, 8, axis=0)
    # tt = int(n_layers / 2)
    #
    # p1 = [n1] * tt
    # p2 = [n2] * (n_layers - tt)
    #
    # latent = p1 + [] + p2
    # res2 = np.repeat(res, 8, axis=0)
    # model.generateTruncated(latent, noi=res2, outImage=True, avg=False, trunc=0.0)

    encoder = UnionEncoder(model)
    x1, x2 = encoder.encode_images(gen)
    images = encoder.transform(x1, x2)
    save_images(images, "../results/styleGAN/inverce_im2")

    plt.figure()
    plt.plot(encoder.loss_hist[0])
    plt.show()
