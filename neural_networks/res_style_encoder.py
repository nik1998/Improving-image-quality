import keras

from utils.mykeras_utils import *


class StyleNet(keras.Model):
    def __init__(self, num_layers=15, im_shape=(None, 128, 128, 1), num_features=64, channels=1, alpha=1.0,
                 style_weight=4.0):
        super(StyleNet, self).__init__()
        self.num_layers = num_layers

        self.loss_fn = None
        self.optimizer = None
        self.alpha = alpha
        self.style_weight = style_weight
        self.num_features = num_features
        self.channels = channels
        self.relu = tf.nn.relu
        self.encoder = self.build_encoder(im_shape)
        # self.encoder.summary()
        self.decoder = self.build_decoder(self.encoder.output_shape)
        # self.decoder.summary()

        self.avg_encoder = self.build_encoder(im_shape)
        self.avg_encoder.set_weights(self.encoder.get_weights())
        self.avg_encoder.trainable = False
        self.avg_decoder = self.build_decoder(self.encoder.output_shape)
        self.avg_decoder.set_weights(self.decoder.get_weights())
        self.avg_decoder.trainable = False
        self.beta = 0.99

    def compile(self, optimizer):
        super(StyleNet, self).compile()
        self.loss_fn = keras.losses.MeanSquaredError()
        self.optimizer = optimizer

    def get_mean_std(self, x, epsilon=1e-5):
        axes = [1, 2]

        # Compute the mean and standard deviation of a tensor.
        mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
        standard_deviation = tf.sqrt(variance + epsilon)
        return mean, standard_deviation

    def ada_in(self, style, content):
        content_mean, content_std = self.get_mean_std(content)
        style_mean, style_std = self.get_mean_std(style)
        t = style_std * (content - content_mean) / content_std + style_mean
        t = self.alpha * t + (1 - self.alpha) * content
        return t

    def build_encoder(self, im_shape):
        conv_layers = []

        conv_layers.append(
            layers.Conv2D(self.num_features, kernel_size=3, strides=2, padding="same",
                          activation=layers.Activation("relu")))
        for i in range(self.num_layers - 1):
            conv_layers.append(layers.Conv2D(self.num_features, kernel_size=3, padding="same",
                                             activation=layers.Activation("relu")))

        input = layers.Input(im_shape)
        x = input
        conv_feats = []
        for i in range(self.num_layers):
            x = conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)
        return keras.Model(input, [x, conv_feats])

    def build_decoder(self, inputs):
        deconv_layers = []
        for i in range(self.num_layers - 1):
            deconv_layers.append(layers.Conv2DTranspose(self.num_features, kernel_size=3, padding="same",
                                                        activation=layers.Activation("relu")))
        deconv_layers.append(
            layers.Conv2DTranspose(self.channels, kernel_size=3, strides=2, padding="same"))

        input = layers.Input(inputs[0][1:])
        x = input
        conv_feats = [layers.Input(i[1:]) for i in inputs[1]]
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)
        x = self.relu(x)
        return keras.Model([input, conv_feats], x)

    def call(self, input_tensor, mask=None, **kwargs):
        pass

    def ema(self):
        for model, avg_model in zip([self.encoder, self.decoder], [self.avg_encoder, self.avg_decoder]):
            # su = 0
            for i in range(len(model.layers)):
                up_weight = model.layers[i].get_weights()
                old_weight = avg_model.layers[i].get_weights()
                new_weight = []
                for j in range(len(up_weight)):
                    # su += tf.reduce_sum(tf.abs(old_weight[j] - up_weight[j]))
                    new_weight.append(old_weight[j] * self.beta + (1 - self.beta) * up_weight[j])
                avg_model.layers[i].set_weights(new_weight)
            # print(su)

    def mainit(self):
        self.avg_encoder.set_weights(self.encoder.get_weights())
        self.avg_decoder.set_weights(self.decoder.get_weights())

    # @tf.function(autograph=not True)
    def train_step(self, input_tensor, **kwargs):
        y, x = input_tensor
        initial_image = x
        # initial_style = y

        with tf.GradientTape(persistent=True) as tape:
            x, conv_feats = self.encoder(x)
            y, style_features = self.encoder(y)
            t = self.ada_in(y, x)

            identity_image = self.decoder([x, conv_feats])
            # identity_style = self.decoder([y, style_features])
            x = self.decoder([t, conv_feats])
            identity_loss = 0.1 * self.loss_fn(identity_image, initial_image)
            # + self.loss_fn(identity_style, initial_style)) / 2
            loss_content, loss_style = self.get_loss(style_features, x, t)
            total_loss = loss_content + self.style_weight * loss_style + identity_loss
        gradients = tape.gradient(total_loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables))
        gradients = tape.gradient(total_loss, self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.decoder.trainable_variables))

        return {
            "identity_loss": identity_loss,
            "style_loss": loss_style,
            "content_loss": loss_content,
            "total_loss": total_loss,
        }

    def test_step(self, input_tensor, **kwargs):
        y, x = input_tensor
        x, conv_feats = self.encoder(x)
        y, style_features = self.encoder(y)
        t = self.ada_in(y, x)
        x = self.decoder([t, conv_feats])
        loss_content, loss_style = self.get_loss(style_features, x, t)
        total_loss = loss_content + self.style_weight * loss_style
        return {
            "style_loss": loss_style,
            "content_loss": loss_content,
            "total_loss": total_loss,
        }

    def get_loss(self, style_features, reconstructed_image, t):
        x, reconstructed_features = self.avg_encoder(reconstructed_image)
        loss_content = self.loss_fn(t, x)
        loss_style = 0.0
        for inp, out in zip(style_features, reconstructed_features):
            mean_inp, std_inp = self.get_mean_std(inp)
            mean_out, std_out = self.get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(std_inp, std_out)
        return loss_content, loss_style

    def predict(self, input_tensor, **kwargs):
        y, x = input_tensor
        x, conv_feats = self.encoder(x)
        y, _ = self.encoder(y)
        t = self.ada_in(y, x)
        x = self.decoder([t, conv_feats])
        return x.numpy()

    def save(self, name, **kwargs):
        self.encoder.save(name + "/enc.h5")
        self.decoder.save(name + "/dec.h5")

    def save_weights(self, name, **kwargs):
        self.encoder.save_weights(name + "/enc.h5")
        self.decoder.save_weights(name + "/dec.h5")

    def load_weights(self, name, **kwargs):
        self.encoder.load_weights(name + "/enc.h5")
        self.decoder.load_weights(name + "/dec.h5")


def style_image_predict(gen, model):
    def f():
        data = gen.next()
        prediction = model.predict(data)
        data = np.concatenate(data, axis=1)
        save_images(np.concatenate([data, prediction], axis=1), path="../results/enc_style/")
        return data[:4], prediction[:4]

    return f


class EncDecAverager(keras.callbacks.Callback):

    def __init__(self, styleNet: StyleNet):
        self.steps = 0
        self.styleNet = styleNet

    def on_train_batch_end(self, batch, logs=None):
        if self.steps <= 100 and self.steps % 10 == 2:
            self.styleNet.mainit()

        if self.steps > 100 and self.steps % 10 == 0:
            self.styleNet.ema()
        self.steps += 1


if __name__ == "__main__":
    batch_size = 4
    style_dir = "../datasets/style"
    content_dir = "../datasets/final_good_images"
    # style_dir = "../datasets/unet/images"
    # content_dir = "../datasets/unet/mask"
    im_size = 256
    im_channels = 1

    train_generator, val_generator = create_image_to_image_generator([style_dir, content_dir],
                                                                     batch_size=batch_size,
                                                                     im_size=im_size,
                                                                     different_seed=True)

    test_generator("../results/test", train_generator, count=100)
    autoencoder = StyleNet(im_shape=(im_size, im_size, im_channels), num_layers=8, channels=im_channels,
                           style_weight=10)
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    autoencoder.compile(optimizer=optimizer)
    autoencoder.build([(None, im_size, im_size, im_channels), (None, im_size, im_size, im_channels)])
    # autoencoder.load_weights("../models/enc_style/model013020:32")
    # autoencoder.summary()
    callbacks = get_callbacks("../models/enc_style",
                              train_pred_func=style_image_predict(val_generator, autoencoder),
                              monitor_loss="val_total_loss", custom_save=True)
    callbacks.append(EncDecAverager(autoencoder))
    t = min(500, len(train_generator))
    v = min(50, len(val_generator))
    history = autoencoder.fit(train_generator, steps_per_epoch=t, epochs=10, validation_data=val_generator,
                              validation_steps=v, callbacks=callbacks)
    plot_graphs(history.history)
