import numpy as np
from tqdm import tqdm

from neural_networks.stylegan import *
from utils.mykeras_utils import plot_graphs
from utils.mylibrary import save_images


def auto_correlation(v):
    return tf.reduce_mean(v * tf.roll(v, shift=1, axis=2)) ** 2 + \
           tf.reduce_mean(v * tf.roll(v, shift=1, axis=1)) ** 2


def calc_reg_loss(noise_input):
    req_loss = 0.0
    for out in noise_input:
        sz = out.shape[1]
        while sz >= 8:
            req_loss += auto_correlation(out)
            # equal average pooling
            out = tf.reshape(out, [-1, sz // 2, 2, sz // 2, 2, 1])
            out = tf.reduce_mean(out, axis=[2, 4])
            sz = sz // 2
    return req_loss


class UnionEncoder:
    def __init__(self, model, steps):
        model.GAN.GMA.trainable = False
        self.model = model
        self.steps = steps
        self.loss_hist = {"img_loss": [], "reg_loss": []}
        self.init_style(model)
        self.initial_noise_factor = 0.0001
        self.noise_ramp_length = 750
        self.regularize_noise_weight = 1e5
        self.regularize_weight = 1
        self.square_loss = tf.keras.losses.MeanSquaredError()

    def init_style(self, model: StyleGAN):
        dlatent_avg_samples = 10000
        latent_samples = noise(dlatent_avg_samples)
        dlatent_samples = model.GAN.SE.predict(latent_samples, batch_size=8)
        self._dlatent_avg = np.mean(dlatent_samples, axis=0, keepdims=True)
        self._dlatent_std = (np.sum((dlatent_samples - self._dlatent_avg) ** 2) / dlatent_avg_samples) ** 0.5

    def encode_images(self, gen):
        res1 = []
        res2 = []
        k = len(gen)
        for _ in tqdm(range(k)):
            self.loss_hist = {"img_loss": [], "reg_loss": []}
            x1, x2 = self._encode_batch(gen.next())
            res1.append(x1)
            res2.append(x2)
        tres1 = np.transpose(np.asarray(res1), [1, 0, 2, 3])
        tres2 = np.transpose(np.asarray(res2), [1, 0, 2, 3, 4, 5])
        return np.squeeze(tres1), np.reshape(tres2, [-1, im_size, im_size, 1])

    def _encode_batch(self, images):
        n = len(images)
        assert n == 1
        n1 = self._dlatent_avg

        x1 = [tf.Variable(n1) for _ in range(7)]
        x2 = [tf.Variable(nImage(n)) for _ in range(7)]
        target = tf.constant(images)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        for t in tqdm(range(self.steps)):
            # noise_strength = self._dlatent_std * self.initial_noise_factor \
            # * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
            # x1t = tf.random.normal([1, latent_size], 0.0, noise_strength)
            # x1.assign(x1 + x1t)
            l = self._opt_step(x1, x2, target)
            for k in self.loss_hist:
                self.loss_hist[k].append(l[k].numpy())
        return [x.numpy() for x in x1], [x.numpy() for x in x2]

    def _opt_step(self, x1, x2, target):
        with tf.GradientTape(persistent=True) as g:
            result = self.model.GAN.GMA(x1 + x2)
            img_loss = tf.reduce_sum(tf.abs(result - target))
            # img_loss = self.square_loss(result, target)
            reg_loss = calc_reg_loss(x2)
            loss = img_loss + self.regularize_noise_weight * reg_loss
            for x in x2:
                loss += tf.reduce_sum(x*x)*self.regularize_weight
        print(loss)
        grad1 = g.gradient(loss, x1)
        self.opt.apply_gradients(zip(grad1, x1))
        grad2 = g.gradient(loss, x2)
        self.opt.apply_gradients(zip(grad2, x2))
        return {"img_loss": img_loss, "reg_loss": reg_loss}

    def transform(self, x1, x2):
        # l = np.repeat(np.expand_dims(x1, axis=0), 7, axis=0)
        l2 = np.reshape(x2, (7, x1.shape[1], im_size, im_size, 1))
        return self.model.GAN.GMA(list(x1) + list(l2)).numpy()


def mixed_style(model, s1, s2, noise):
    tt = int(n_layers / 2)
    p1 = [s1] * tt
    p2 = [s2] * (n_layers - tt)
    latent = p1 + [] + p2
    im = model.GAN.GMA(latent + list(noise)).numpy()
    save_images(im, "../results/styleGAN_inverse/style_merge")


def generateTruncated(model, style, noi, trunc=0.5, outImage=False, avg=False, num=0, rim=8):
    if avg:
        av = np.mean(model.GAN.S.predict(noise(2000), batch_size=64), axis=0, keepdims=True)
    else:
        av = np.zeros((1, latent_size))

    w_space = []
    for i in range(len(style)):
        tempStyle = model.GAN.SE.predict(style[i])
        tempStyle = (1 - trunc) * tempStyle + trunc * av
        w_space.append(tempStyle)
    generated_images = model.GAN.GE.predict(w_space + noi, batch_size=BATCH_SIZE)

    if outImage:
        concat_clip_save(generated_images, "../results/styleGAN_inverse/inv_images/t" + str(num) + ".png", rim)

    return generated_images


if __name__ == "__main__":
    aug = AugmentationUtils() \
        .rescale()

    batch_size = 1
    gen = aug.create_generator("../datasets/test_sem_internet",
                               target_size=(im_size, im_size),
                               batch_size=batch_size,
                               color_mode='grayscale',
                               class_mode=None)
    model = StyleGAN(inverse=True)
    # model.save(0)
    model.load(26)
    image_dir = "../results/styleGAN_inverse/inv_images"

    encoder = UnionEncoder(model, 1000)
    x1, x2 = encoder.encode_images(gen)
    # np.save(output + 'best.npy', best_latent)
    save_images(x2, image_dir)
    images = encoder.transform(x1, x2)
    save_images(images, image_dir)

    plot_graphs(encoder.loss_hist)

    # tx = np.reshape(x2, (7, len(x1), im_size, im_size, 1))
    # mixed_style(model, x1[:1], x1[1:], tx[:, :1])
    # mixed_style(model, x1[1:], x1[:1], tx[:, 1:])
    #
    # dataset_len = len(x1)
    #
    # n1 = np.tile(x1, (dataset_len, 1))
    # n2 = np.repeat(x1, dataset_len, axis=0)
    # tt = int(n_layers // 2)
    #
    # p1 = [n1] * tt
    # p2 = [n2] * (n_layers - tt)
    #
    # latent = p1 + [] + p2
    # res2 = list(np.repeat(tx, dataset_len, axis=1))
    # generateTruncated(model, latent, noi=res2, outImage=True, avg=False, trunc=0.0, num=0, rim=dataset_len)
    # generateTruncated(model, latent, noi=res2, outImage=True, avg=True, trunc=0.0, num=1, rim=dataset_len)
    # generateTruncated(model, latent, noi=res2, outImage=True, avg=True, trunc=0.5, num=2, rim=dataset_len)
