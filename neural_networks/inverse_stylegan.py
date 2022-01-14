import cv2
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation

from neural_networks.stylegan import *


class OptimisationEncoder:
    def __init__(self, model, batch_size=16):
        self.model = model
        self.batch_size = batch_size
        self.steps = [(0.1, 200), (0.05, 100), (0.01, 100), (0.005, 100)]
        self.loss_hist = []
        self.batch_size = batch_size
        self.n1 = noiseList(1)

    def _opt_step(self, x1, x2, target, lr):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA(n_layers * [x1] + [x2])
            loss = tf.reduce_sum(tf.abs(result - target))
        dx2 = g.gradient(loss, x2)
        # x1.assign(x1 - lr * dx1)
        x2.assign(x2 - lr * dx2)
        return loss

    def _encode_batch(self, images):
        n = len(images)
        n1 = np.repeat(self.n1[0], n, axis=0)
        x1 = tf.Variable(n1)
        n2 = np.random.uniform(0.0, 1.0, size=[n, im_size, im_size, 1]).astype('float32')
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
        i = 0
        res = []
        k = len(gen)
        while i < k:
            next_i = min(k, i + self.batch_size)
            res.append(self._encode_batch(gen.next()))
            i = next_i
            print(next_i / k)
        return np.concatenate(res)

    def transform(self, lattents):
        images = []
        for lattent in lattents:
            n2 = tf.expand_dims(lattent, 0)
            images.append(self.model.GAN.GMA(self.n1 + [n2])[0].numpy())
        return images


if __name__ == "__main__":
    aug = AugmentationUtils() \
        .rescale()

    gen = aug.create_generator("../datasets/test_sem_internet",
                               target_size=(im_size, im_size),
                               batch_size=1,
                               color_mode='rgb',
                               class_mode=None)
    model = StyleGAN(gen, lr=0.0001, silent=False)
    #model.save(100)
    model.load(8)

    encoder = OptimisationEncoder(model, 1)

    res = encoder.encode_images(gen)

    images = encoder.transform(res)

    plt.plot(encoder.loss_hist[0])
    plt.show()

    save_images(images, "../results/styleGAN/testim")
