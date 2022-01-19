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
        l = tf.repeat(self.input_as_weights, len(input_), axis=1)
        return self.model.GAN.GMA([l] * n_layers + [input_])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'model': None,
            'init_value': self.init_value,
        })
        return config


def grad_descent_find_face():
    batch_size = 8
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

    train_generator = UnionGenerator([t_gen, t_gen2], batch_size).reflect_rotate()
    # test_generator("../results/test", train_generator)
    val_generator = UnionGenerator([v_gen, v_gen2], batch_size).reflect_rotate()
    # test_generator("../results/test/val", train_generator)

    style_model = StyleGAN(None)
    style_model.load(5)

    model = tf.keras.Sequential([
        StyleGANLayer(style_model, noise(1), name="style", input_shape=(im_size, im_size, im_channel)),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MSE)

    callbacks = [
        keras.callbacks.ModelCheckpoint("../models/style-inv/best_model.h5", monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=5, cooldown=15, verbose=1, min_lr=1e-4)
    ]
    hist = model.fit(train_generator, epochs=50, callbacks=callbacks, validation_data=val_generator)
    plot_graphs(hist.history)
    model.load_weights("../models/style-inv/best_model.h5")
    w = model.get_layer("style").trainable_variables
    np.save("../results/styleGAN_inverse/inv_styles/first.npy", w[0].numpy())

    pred = model.predict(train_generator)
    save_images(pred, "../results/styleGAN_inverse/inv_images")


if __name__ == '__main__':
    grad_descent_find_face()
