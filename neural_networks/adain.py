from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from utils.mykeras_utils import *


def get_mean_std(x, epsilon=1e-5):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation


def ada_in(style, content, alpha=1.0):
    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    t = alpha * t + (1 - alpha) * content
    return t


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor[-1], input_tensor[-1])
    input_shape = tf.shape(input_tensor[-1])
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def gram_style_loss(base_style, target_style):
    gram_style = gram_matrix(base_style)
    gram_style2 = gram_matrix(target_style)
    return tf.reduce_mean(tf.square(gram_style - gram_style2))


class NeuralStyleTransfer(tf.keras.Model):
    def __init__(self, initial_img_size, style_weight=4, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = None
        self.optimizer = None
        self.kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self._get_style_vgg(initial_img_size)
        self._get_decoder(self.encoder.output)
        self.style_weight = style_weight
        self.alpha = alpha
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def _get_style_vgg(self, input_img_size):
        vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=input_img_size)
        vgg19.trainable = False
        layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
        outputs = [vgg19.get_layer(name).output for name in layer_names]
        self.encoder = keras.Model(vgg19.input, outputs)
        # self.encoder.summary()

    def _get_decoder(self, enc_out):
        config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu",
                  "kernel_initializer": self.kernel_init}
        inputs = [layers.Input(i.shape[1:]) for i in enc_out]
        x = layers.Conv2D(filters=512, **config)(inputs[-1])
        x = layers.Add()([x, inputs[-1]])
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters=256, **config)(x)
        x = layers.Add()([x, inputs[-2]])
        x = layers.Conv2D(filters=256, **config)(x)
        x = layers.Conv2D(filters=256, **config)(x)
        x = layers.Conv2D(filters=256, **config)(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters=128, **config)(x)
        x = layers.Add()([x, inputs[-3]])
        x = layers.Conv2D(filters=128, **config)(x)
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(filters=64, **config)(x)
        x = layers.Add()([x, inputs[-4]])
        x = layers.Conv2D(filters=3, kernel_size=3, strides=1, padding="same", activation="sigmoid")(x)
        self.decoder = keras.Model(inputs, x)
        # self.decoder.summary()

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, input_tensor, mask=None, **kwargs):
        pass

    def predict(self, input_tensor):
        content, style = input_tensor
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)
        for i in range(len(content_encoded)):
            content_encoded[i] = ada_in(style=style_encoded[i], content=content_encoded[i], alpha=self.alpha)
        return self.decoder(content_encoded)

    def get_losses(self, inputs):
        content, style = inputs

        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)
        identity_loss = self.identity_loss_fn(self.decoder(content_encoded), content) * 100

        # Compute the AdaIN target feature maps.
        for i in range(len(content_encoded)):
            content_encoded[i] = ada_in(style=style_encoded[i], content=content_encoded[i], alpha=self.alpha)

        # Generate the neural style transferred image.
        reconstructed_image = self.decoder(content_encoded)
        # identity_loss += self.identity_loss_fn(reconstructed_image[..., 0], reconstructed_image[..., 1]) * 10
        # identity_loss += self.identity_loss_fn(reconstructed_image[..., 1], reconstructed_image[..., 2]) * 10
        # Compute the losses.
        reconstructed_vgg_features = self.encoder(reconstructed_image)
        content_loss = self.loss_fn(content_encoded[-1], reconstructed_vgg_features[-1])
        style_loss = 0.0
        for inp, out in zip(style_encoded, reconstructed_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            style_loss += self.loss_fn(mean_inp, mean_out) + self.loss_fn(std_inp, std_out)
        style_loss = self.style_weight * style_loss
        total_loss = content_loss + style_loss + identity_loss
        return {
            "style_loss": style_loss,
            "content_loss": content_loss,
            "identity_loss": identity_loss,
            "total_loss": total_loss,
        }

    # @tf.function(autograph=not True)
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            losses = self.get_losses(inputs)
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(losses["total_loss"], trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return losses

    def test_step(self, inputs):
        return self.get_losses(inputs)

    def save_weights(self, name, **kwargs):
        self.decoder.save_weights(name)

    def load_weights(self, name, **kwargs):
        self.decoder.load_weights(name)


def style_image_predict(gen, model):
    def f():
        data = gen.next()
        prediction = model.predict(data).numpy()
        data = np.concatenate(data, axis=1)
        save_images(np.concatenate([data, prediction], axis=1), path="../results/style/adain/")
        return data[:4], prediction[:4]

    return f


def train():
    lr = 1e-4
    lr_decay = 5e-6
    im_size = 256
    batch_size = 8
    style_weight = 4
    initial_img_size = (im_size, im_size, 3)
    alpha = 0.1
    epoch = 50

    style_dir = "../datasets/style"
    content_dir = '../datasets/unet/small_scale/all_real_images'

    train_generator, val_generator = create_image_to_image_generator([content_dir, style_dir],
                                                                     batch_size=batch_size,
                                                                     im_size=im_size,
                                                                     color_mode='rgb',
                                                                     different_seed=True)

    test_generator("../results/test", train_generator, 200)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.MeanSquaredError()
    model = NeuralStyleTransfer(initial_img_size, style_weight=style_weight, alpha=alpha)

    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    callbacks = get_callbacks('../models/adain/', train_pred_func=style_image_predict(val_generator, model),
                              monitor_loss='val_total_loss')

    callbacks.append(ReduceLROnPlateau(monitor='val_total_loss', factor=0.2,
                                       patience=5, cooldown=15, verbose=1, min_lr=lr_decay))
    history = model.fit(train_generator, epochs=epoch, validation_data=val_generator, callbacks=callbacks)
    plot_graphs(history.history)

    images = get_gen_images(val_generator, 8)
    stylized_images = model.predict(images)
    images = rgb_to_gray(images[0])
    stylized_images = rgb_to_gray(stylized_images)
    unionTestImages(images, stylized_images, path="../results/style/adain")


if __name__ == "__main__":
    train()
