import math

from utils.mykeras_utils import *
from keras import layers


class REDNet(keras.Model):
    def __init__(self, num_layers=15, num_features=64, channels=1):
        super(REDNet, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(
            layers.Conv2D(num_features, kernel_size=3, strides=2, padding="same",
                          activation=layers.Activation("relu")))

        for i in range(num_layers - 1):
            conv_layers.append(layers.Conv2D(num_features, kernel_size=3, padding="same",
                                             activation=layers.Activation("relu")))

        for i in range(num_layers - 1):
            deconv_layers.append(layers.Conv2DTranspose(num_features, kernel_size=3, padding="same",
                                                        activation=layers.Activation("relu")))
        self.final_dec = layers.Conv2DTranspose(channels, kernel_size=3, strides=2, padding="same")

        # self.final_dec = layers.Conv2D(channels, kernel_size=3, padding="same", kernel_initializer='he_uniform',
        #                                activation=layers.Activation("relu"))
        # self.final_dec2 = layers.Conv2D(channels, kernel_size=3, padding="same", kernel_initializer='he_uniform',
        #                                 activation=layers.Activation("relu"))

        self.reflect_padding = ReflectionPadding2D()
        self.conv_layers = conv_layers
        self.deconv_layers = deconv_layers
        self.relu = tf.nn.relu

    def call(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x = self.final_dec(x)
        x += residual
        x = self.relu(x)
        # x = self.reflect_padding(x)
        # x = self.final_dec2(x)
        return x


def noise_function(aug: AugmentationUtils):
    return aug.add_defect_expansion_algorithm(gauss=False)  # .random_gauss_noise()


def loss_maxd_fn(y_true, y_pred):
    return tf.reduce_max(tf.abs(y_true - y_pred))


class TotalLoss:
    def __init__(self, w1, w2):
        self.content_loss = keras.losses.MeanSquaredError()
        self.maxd_loss = loss_maxd_fn
        self.w1 = w1
        self.w2 = w2

    def __call__(self, y_true, y_pred, **kwargs):
        return self.w1 * self.content_loss(y_true, y_pred) + self.w2 * self.maxd_loss(y_true, y_pred)


if __name__ == "__main__":
    batch_size = 1
    # image_dir = '../datasets/not_sem/cats/real'
    image_dir = "../datasets/imagenet/style"
    im_size = 256
    im_channels = 1

    train_generator, val_generator = create_image_to_image_generator([image_dir, image_dir],
                                                                     aug_extension=[noise_function],
                                                                     batch_size=batch_size,
                                                                     im_size=im_size, vertical_flip=False,
                                                                     ninty_rotate=False)

    autoencoder = REDNet(num_layers=5, channels=im_channels)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    loss = TotalLoss(2, 4)

    autoencoder.compile(optimizer=optimizer, loss=loss)
    autoencoder.build((None, im_size, im_size, 1))
    autoencoder.summary()
    # test_generator("../results/test", train_generator, 1000)
    callbacks = get_default_callbacks("../models/rednetimp", train_generator, autoencoder)

    history = autoencoder.fit(train_generator, epochs=200, validation_data=val_generator, callbacks=callbacks)

    plot_graphs(history.history)

    images = get_gen_images(val_generator, 100)

    autoencoder.load_weights(get_latest_filename("../models/rednetimp"))
    test = autoencoder.predict(images[0])

    unionTestImages(images[0], test, path="../results/enc_dec_simple2")
    save_images(np.concatenate([images[0], test], axis=2), path="../results/enc_dec_simple2")

    images = get_gen_images(train_generator)
    test = autoencoder.predict(images[0])
    save_images(np.concatenate([images[0], test], axis=2), path="../results/enc_dec_simple2")
