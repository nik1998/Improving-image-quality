import numpy as np

from neural_networks.stylegan import *
from utils.mykeras_utils import *

tf.compat.v1.enable_eager_execution()


class StyleGANLayer(tf.keras.layers.Layer):

    def __init__(self, model, init_value, **kwargs):
        super(StyleGANLayer, self).__init__(**kwargs)
        self.init_value = init_value
        self.model = model

    def build(self, input_shape):
        if self.init_value is None:
            initer = tf.initializers.random_normal(0, 1)
        else:
            initer = tf.constant_initializer(self.init_value)
        self.input_as_weights = self.add_weight("input_as_weights",
                                                shape=(1, 512), trainable=True, initializer=initer)

    def call(self, input_):
        l = tf.repeat(self.input_as_weights, len(input_), axis=0)
        return self.model.GAN.GMA([l] * n_layers + [input_])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model': None,
            'init_value': self.init_value,
        })
        return config


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, gan_model, gen, num=2, single=None):
        self.gen = gen
        self.model = gan_model
        self.img_num = num
        self.single = single

    def on_epoch_end(self, epoch, logs=None):
        if self.single is not None:
            imgs, out = self.single
        else:
            imgs, out = self.gen.next()
            imgs = imgs[:self.img_num]
            out = out[:self.img_num]
        prediction = self.model(imgs, out)
        for i, (img, pred) in enumerate(zip(out, prediction)):
            c = np.concatenate([img, pred])
            showImage(c)
            cv2.imwrite("./results/styleGAN_inverse/generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1), c)


def grad_descent_find_image():
    batch_size = 1
    seed = random.randint(0, 2 ** 30)

    aug = AugmentationUtils().rescale(). \
        zoom_range(0.7, 1). \
        horizontal_flip(). \
        vertical_flip(). \
        validation_split()

    t_gen = aug.create_generator("../datasets/one_layer_images/splited1",
                                 target_size=(im_size, im_size),
                                 batch_size=batch_size,
                                 class_mode=None,
                                 subset="training",
                                 seed=seed)

    t_gen2 = aug.create_generator("../datasets/cycle/sem_to_sem/imgsA",
                                  target_size=(im_size, im_size),
                                  batch_size=batch_size,
                                  class_mode=None,
                                  subset="training",
                                  seed=seed)

    v_gen = aug.create_generator("../datasets/one_layer_images/splited1",
                                 target_size=(im_size, im_size),
                                 batch_size=batch_size,
                                 class_mode=None,
                                 subset="validation",
                                 seed=seed)

    v_gen2 = aug.create_generator("../datasets/cycle/sem_to_sem/imgsA",
                                  target_size=(im_size, im_size),
                                  batch_size=batch_size,
                                  class_mode=None,
                                  subset="validation",
                                  seed=seed)

    train_generator = UnionGenerator([t_gen, t_gen2]).ninty_rotate()
    # test_generator("../results/test", train_generator)
    val_generator = UnionGenerator([v_gen, v_gen2]).ninty_rotate()
    # test_generator("../results/test/val", train_generator)

    style_model = StyleGAN(None)
    style_model.load(20)
    n = noise(1)
    model = tf.keras.Sequential([
        StyleGANLayer(style_model, n, name="style", input_shape=(im_size, im_size, im_channel)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MSE)
    # model.summary()
    x, y = train_generator.next()
    x = x[:1]
    y = y[:1]
    zx = np.zeros_like(x)

    callbacks = [
        keras.callbacks.ModelCheckpoint("../models/style-inv/best_model.h5", monitor="loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                          patience=5, cooldown=15, verbose=1, min_lr=1e-6),
        GANMonitor(model, gen=None, single=(zx, y))
    ]
    hist = model.fit(zx, y, epochs=100, callbacks=callbacks)
    plot_graphs(hist.history)
    model.load_weights("../models/style-inv/best_model.h5")
    w = model.get_layer("style").trainable_variables
    w = w[0].numpy()
    np.save("../results/styleGAN_inverse/inv_styles/first.npy", w)

    ws = np.repeat(w, 64, axis=0)
    # [ws] * n_layers
    res = style_model.GAN.GMA(noiseList(64) + [nImage(64)])
    concat_clip_save(res, "../results/styleGAN_inverse/inv_images/test.png", 8)
    print(np.sum(np.abs(n - w)))
    # pred = model.predict(train_generator)
    # save_images(pred, "../results/styleGAN_inverse/inv_images")


if __name__ == '__main__':
    grad_descent_find_image()
