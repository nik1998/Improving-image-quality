from typing import Sequence, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils.mykeras_utils import create_image_to_image_generator, AugmentationUtils, get_default_callbacks, \
    get_gen_images
from utils.mylibrary import plot_graphs, save_images


# https://github.com/dslisleedh/NAFNet-tensorflow2/blob/main/layers.py

def edge_padding2d(x, h_pad, w_pad):
    if h_pad[0] != 0:
        x_up = tf.gather(x, indices=[0], axis=1)
        x_up = tf.concat([x_up for _ in range(h_pad[0])], axis=1)
        x = tf.concat([x_up, x], axis=1)
    if h_pad[1] != 0:
        x_down = tf.gather(tf.reverse(x, axis=[1]), indices=[0], axis=1)
        x_down = tf.concat([x_down for _ in range(h_pad[1])], axis=1)
        x = tf.concat([x, x_down], axis=1)
    if w_pad[0] != 0:
        x_left = tf.gather(x, indices=[0], axis=2)
        x_left = tf.concat([x_left for _ in range(w_pad[0])], axis=2)
        x = tf.concat([x_left, x], axis=2)
    if w_pad[1] != 0:
        x_right = tf.gather(tf.reverse(x, axis=[2]), indices=[0], axis=2)
        x_right = tf.concat([x_right for _ in range(w_pad[1])], axis=2)
        x = tf.concat([x, x_right], axis=2)
    return x


class LocalAvgPool2D(tf.keras.layers.Layer):
    def __init__(
            self, local_size: Sequence[int]
    ):
        super(LocalAvgPool2D, self).__init__()
        self.local_size = local_size

    def call(self, inputs, training):
        if training:
            return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)

        _, h, w, _ = inputs.get_shape().as_list()
        kh = min(h, self.local_size[0])
        kw = min(w, self.local_size[1])
        inputs = tf.pad(inputs,
                        [[0, 0],
                         [1, 0],
                         [1, 0],
                         [0, 0]]
                        )
        inputs = tf.cumsum(tf.cumsum(inputs, axis=2), axis=1)
        s1 = tf.slice(inputs,
                      [0, 0, 0, 0],
                      [-1, kh, kw, -1]
                      )
        s2 = tf.slice(inputs,
                      [0, 0, (w - kw) + 1, 0],
                      [-1, kw, -1, -1]
                      )
        s3 = tf.slice(inputs,
                      [0, (h - kh) + 1, 0, 0],
                      [-1, -1, kw, -1]
                      )
        s4 = tf.slice(inputs,
                      [0, (h - kh) + 1, (w - kw) + 1, 0],
                      [-1, -1, -1, -1]
                      )
        local_ap = (s4 + s1 - s2 - s3) / (kh * kw)

        _, h_, w_, _ = local_ap.get_shape().as_list()
        h_pad, w_pad = [(h - h_) // 2, (h - h_ + 1) // 2], [(w - w_) // 2, (w - w_ + 1) // 2]
        local_ap = edge_padding2d(local_ap, h_pad, w_pad)
        return local_ap


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, upsample_rate):
        super(PixelShuffle, self).__init__()
        self.upsample_rate = upsample_rate

    def call(self, inputs, *args, **kwargs):
        return tf.nn.depth_to_space(
            inputs, block_size=self.upsample_rate
        )


class SimpleGate(tf.keras.layers.Layer):
    def __init__(self):
        super(SimpleGate, self).__init__()

    def call(self, inputs, *args, **kwargs):
        x1, x2 = tf.split(
            inputs, num_or_size_splits=2, axis=-1
        )
        return x1 * x2


class SimpleChannelAttention(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, kh: int, kw: int
    ):
        super(SimpleChannelAttention, self).__init__()
        self.n_filters = n_filters
        self.kh = kh
        self.kw = kw

        self.pool = LocalAvgPool2D((kh, kw))
        self.w = tf.keras.layers.Dense(
            self.n_filters, activation=None
        )

    def call(self, inputs, *args, **kwargs):
        attention = self.pool(inputs)
        attention = self.w(attention)
        return attention * inputs


class NAFBlock(tf.keras.layers.Layer):
    def __init__(
            self, n_filters: int, dropout_rate: float, kh: int,
            kw: int, dw_expansion: int = 2, ffn_expansion: int = 2
    ):
        super(NAFBlock, self).__init__()
        self.n_filters = n_filters
        self.dropout_rate = dropout_rate
        self.kh = kh
        self.kw = kw
        self.dw_filters = n_filters * dw_expansion
        self.ffn_filters = n_filters * ffn_expansion

        self.spatial = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv2D(
                self.dw_filters, kernel_size=1, strides=1, padding='VALID',
                activation=None
            ),
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=3, strides=1, padding='SAME', activation=None
            ),
            SimpleGate(),
            SimpleChannelAttention(
                self.n_filters, self.kh, self.kw
            ),
            tf.keras.layers.Conv2D(
                self.n_filters, kernel_size=1, strides=1, padding='VALID',
                activation=None
            )
        ])
        self.spatial_drop = tf.keras.layers.Dropout(self.dropout_rate)

        self.channel = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(self.ffn_filters,
                                  activation=None
                                  ),
            SimpleGate(),
            tf.keras.layers.Dense(self.n_filters,
                                  activation=None
                                  )
        ])
        self.channel_drop = tf.keras.layers.Dropout(self.dropout_rate)

        self.beta = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)),
            trainable=True,
            dtype=tf.float32
        )
        self.gamma = tf.Variable(
            tf.zeros((1, 1, 1, self.n_filters)),
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, *args, **kwargs):
        inputs = self.spatial_drop(self.spatial(inputs)) * self.beta + inputs
        inputs = self.channel_drop(self.channel(inputs)) * self.gamma + inputs
        return inputs


class NAFNet(tf.keras.models.Model):
    def __init__(
            self, width: int = 16, n_middle_blocks: int = 1, n_enc_blocks: Sequence[int] = (1, 1, 1, 28),
            n_dec_blocks: Sequence[int] = (1, 1, 1, 1), dropout_rate: float = 0.,
            train_size: Sequence[Optional[int]] = (None, 256, 256, 3), tlsc_rate: float = 1.5
    ):
        super(NAFNet, self).__init__()
        self.width = width
        self.n_middle_blocks = n_middle_blocks
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.dropout_rate = dropout_rate
        self.train_size = train_size
        self.tlsc_rate = tlsc_rate
        n_stages = len(n_enc_blocks)
        kh, kw = int(train_size[1] * tlsc_rate), int(train_size[2] * tlsc_rate)

        self.to_features = tf.keras.layers.Conv2D(
            width, kernel_size=3, padding='SAME', activation=None,
            strides=1
        )
        self.to_rgb = tf.keras.layers.Conv2D(
            3, kernel_size=3, padding='SAME', activation=None,
            strides=1
        )
        self.encoders = []
        self.downs = []
        for i, n in enumerate(n_enc_blocks):
            self.encoders.append(
                tf.keras.Sequential([
                    NAFBlock(
                        width * (2 ** i), dropout_rate, kh // (2 ** i), kw // (2 ** i)
                    ) for _ in range(n)
                ])
            )
            self.downs.append(
                tf.keras.layers.Conv2D(
                    width * (2 ** (i + 1)), kernel_size=2, padding='valid', strides=2,
                    activation=None
                )
            )
        self.middles = tf.keras.Sequential([
            NAFBlock(
                width * (2 ** n_stages), dropout_rate, kh // (2 ** n_stages), kw // (2 ** n_stages)
            ) for _ in range(n_middle_blocks)
        ])
        self.decoders = []
        self.ups = []
        for i, n in enumerate(n_dec_blocks):
            self.ups.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(
                        width * (2 ** (n_stages - i)) * 2, kernel_size=1, padding='VALID', activation=None,
                        strides=1
                    ),
                    PixelShuffle(2)
                ])
            )
            self.decoders.append(
                tf.keras.Sequential([
                    NAFBlock(
                        width * (2 ** (n_stages - (i + 1))), dropout_rate,
                        kh // (2 ** (n_stages - (i + 1))), kw // (2 ** (n_stages - (i + 1)))
                    ) for _ in range(n)
                ])
            )

    @tf.function
    def forward(self, x, training=False):
        features = self.to_features(x, training=training)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            features = encoder(features, training=training)
            encs.append(features)
            features = down(features, training=training)

        features = self.middles(features, training=training)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            features = up(features, training=training)
            features = features + enc_skip
            features = decoder(features, training=training)

        x_res = self.to_rgb(features, training=training)
        x = x + x_res
        return x

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = False
        return self.forward(inputs, training=training)


def noise_function(aug: AugmentationUtils):
    return aug.add_defect_expansion_algorithm(gauss=False)  # .random_gauss_noise()


def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


if __name__ == "__main__":
    image_dir = "../datasets/imagenet/style"
    im_size = 256
    batch_size = 4
    train_generator, val_generator = create_image_to_image_generator([image_dir, image_dir],
                                                                     aug_extension=[noise_function],
                                                                     batch_size=batch_size, color_mode='rgb',
                                                                     im_size=im_size, vertical_flip=False,
                                                                     ninty_rotate=False)

    model = NAFNet()
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss=charbonnier_loss, metrics=[peak_signal_noise_ratio]
    )

    callbacks = get_default_callbacks("../models/NAFNet", train_generator, model)
    history = model.fit(
        train_generator,
        #validation_data=val_generator,
        epochs=50,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_peak_signal_noise_ratio",
                factor=0.5,
                patience=5,
                verbose=1,
                min_delta=1e-7,
                mode="max",
            ), callbacks[0]
        ],
    )

    model.summary()

    plot_graphs(history.history)

    images = get_gen_images(val_generator)
    test = model.predict(images[0])
    save_images(np.concatenate([images[0], test], axis=2), path="../results/NAFNet/")
