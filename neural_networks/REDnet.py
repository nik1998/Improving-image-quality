import math

from utils.mykeras_utils import *


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
        deconv_layers.append(
            layers.Conv2DTranspose(channels, kernel_size=3, strides=2, padding="same"))

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

        x += residual
        x = self.relu(x)
        return x


def noise_function(aug: AugmentationUtils):
    return aug.add_gauss_noise()


if __name__ == "__main__":
    batch_size = 32
    image_dir = "../datasets/final_good_images"
    im_size = 256
    im_channels = 1

    train_generator, val_generator = create_image_to_image_generator([image_dir, image_dir],
                                                                     aug_extension=[noise_function],
                                                                     batch_size=batch_size,
                                                                     im_size=im_size)

    test_generator("../results/test", train_generator, 100)

    autoencoder = REDNet(num_layers=5, channels=im_channels)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    loss_fn = keras.losses.MeanSquaredError()
    autoencoder.compile(optimizer=optimizer, loss=loss_fn)
    autoencoder.build((None, im_size, im_size, 1))
    autoencoder.summary()
    autoencoder.load_weights('../models/rednet/model020122:19.h5')

    images = read_dir("../datasets/scan_images/", 0, 0)
    for i in range(len(images)):
        images[i] = gauss_noise(images[i], mean=0, var=0.1)
    save_images(images, '../results/scan_results/')
    process_real_frame(autoencoder, images)


    # autoencoder.summary()
    # autoencoder.load_weights("../models/rednet/model013020:09.h5")

    callbacks = get_default_callbacks("../models/rednet", train_generator, autoencoder)

    history = autoencoder.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callbacks)

    plot_graphs(history.history)

    images = get_gen_images(val_generator, 100)

    test = autoencoder.predict(images[0])

    unionTestImages(images[0], test, path="../results/enc_dec_simple")
