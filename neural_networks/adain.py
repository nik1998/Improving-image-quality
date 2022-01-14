from datetime import datetime

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from utils.mykeras_utils import *

log_dir = "../models/arbitary2/enc_dec/model"
lr = 1e-4
lr_decay = 5e-5
image_size = 128
batch_size = 8
style_weight = 4
orig_img_size = (image_size, image_size)
alpha = 1.0
epoch = 150

style_dir = "../datasets/unet/images"
content_dir = "../datasets/unet/mask"


def get_encoder():
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*orig_img_size, 3),
    )
    vgg19.trainable = False
    mini_vgg19 = keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    inputs = layers.Input([*orig_img_size, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")


def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean and standard deviation of a tensor.
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation


def ada_in(style, content, alpha=1.0):
    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)
    t = style_std * (content - content_mean) / content_std + style_mean
    t = alpha * t + (1 - alpha) * content
    return t


def get_decoder():
    config = {"kernel_size": 3, "strides": 1, "padding": "same", "activation": "relu"}
    decoder = keras.Sequential(
        [
            layers.InputLayer((None, None, 512)),
            layers.Conv2D(filters=512, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.Conv2D(filters=256, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=128, **config),
            layers.Conv2D(filters=128, **config),
            layers.UpSampling2D(),
            layers.Conv2D(filters=64, **config),
            layers.Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="sigmoid",
            ),
        ]
    )
    return decoder


def get_style_vgg():
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet", input_shape=(*orig_img_size, 3)
    )
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(vgg19.input, outputs)

    inputs = layers.Input([*orig_img_size, 3])
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="loss_net")


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
    def __init__(self, encoder, decoder, loss_net, style_weight, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_net = loss_net
        self.style_weight = style_weight
        self.alpha = alpha

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def call(self, input_tensor, mask=None, **kwargs):
        pass

    def predict(self, input_tensor, **kwargs):
        style, content = input_tensor
        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN target feature maps.
        t = ada_in(style=style_encoded, content=content_encoded, alpha=self.alpha)
        reconstructed_image = self.decoder(t)
        return reconstructed_image

    def train_step(self, inputs):
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        with tf.GradientTape() as tape:
            # Encode the style and content image.
            style_encoded = self.encoder(style)
            content_encoded = self.encoder(content)

            # Compute the AdaIN target feature maps.
            t = ada_in(style=style_encoded, content=content_encoded, alpha=self.alpha)

            # Generate the neural style transferred image.
            reconstructed_image = self.decoder(t)

            # Compute the losses.
            reconstructed_vgg_features = self.loss_net(reconstructed_image)
            style_vgg_features = self.loss_net(style)
            loss_content = self.loss_fn(t, reconstructed_vgg_features[-1])
            for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
                mean_inp, std_inp = get_mean_std(inp)
                mean_out, std_out = get_mean_std(out)
                loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(std_inp, std_out)
            loss_style = self.style_weight * loss_style
            total_loss = loss_content + loss_style

        # Compute gradients and optimize the decoder.
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    def test_step(self, inputs):
        style, content = inputs

        # Initialize the content and style loss.
        loss_content = 0.0
        loss_style = 0.0

        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN target feature maps.
        t = ada_in(style=style_encoded, content=content_encoded, alpha=self.alpha)

        # Generate the neural style transferred image.
        reconstructed_image = self.decoder(t)

        # Compute the losses.
        recons_vgg_features = self.loss_net(reconstructed_image)
        style_vgg_features = self.loss_net(style)
        loss_content = self.loss_fn(t, recons_vgg_features[-1])
        for inp, out in zip(style_vgg_features, recons_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                std_inp, std_out
            )
        loss_style = self.style_weight * loss_style
        total_loss = loss_content + loss_style

        # Update the trackers.
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)
        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }

    def get_model(self):
        inputs1 = layers.Input([*orig_img_size, 3])
        inputs2 = layers.Input([*orig_img_size, 3])
        input1 = self.encoder(inputs1)
        input2 = self.encoder(inputs2)
        t = ada_in(style=input1, content=input2, alpha=self.alpha)
        x = self.decoder(t)
        return keras.Model([inputs1, inputs2], x)

    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker,
        ]


class TrainMonitor(tf.keras.callbacks.Callback):

    def __init__(self, test_style, test_content, alpha=1.0):
        self.test_style = test_style
        self.test_content = test_content
        self.alpha = alpha

    def on_epoch_end(self, epoch, logs=None):
        test_style, test_content = self.test_style, self.test_content
        # Encode the style and content image.
        test_style_encoded = self.model.encoder(test_style)
        test_content_encoded = self.model.encoder(test_content)

        # Compute the AdaIN features.
        test_t = ada_in(style=test_style_encoded, content=test_content_encoded, alpha=self.alpha)
        test_reconstructed_image = self.model.decoder(test_t)

        # Plot the Style, Content and the NST image.
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(tf.keras.preprocessing.image.array_to_img(test_style[0]))
        ax[0].set_title(f"Style: {epoch:03d}")

        ax[1].imshow(tf.keras.preprocessing.image.array_to_img(test_content[0]))
        ax[1].set_title(f"Content: {epoch:03d}")

        ax[2].imshow(tf.keras.preprocessing.image.array_to_img(test_reconstructed_image[0]))
        ax[2].set_title(f"NST: {epoch:03d}")

        plt.show()
        plt.close()


if __name__ == "__main__":
    aug = AugmentationUtils() \
        .rescale() \
        .horizontal_flip() \
        .vertical_flip() \
        .reflect_rotation() \
        .validation_split()
    train_content, val_content = aug.train_val_generator(content_dir,
                                                         color_mode="rgb",
                                                         target_size=orig_img_size,
                                                         batch_size=batch_size)
    # .add_median_blur() \
    #   .add_gaussian_blur() \
    aug = AugmentationUtils() \
        .rescale() \
        .horizontal_flip() \
        .vertical_flip() \
        .reflect_rotation() \
        .validation_split()
    train_style, val_style = aug.train_val_generator(style_dir,
                                                     color_mode="rgb",
                                                     target_size=orig_img_size,
                                                     batch_size=batch_size)

    train_generator = UnionGenerator([train_style, train_content], batch_size)
    val_generator = UnionGenerator([val_style, val_content], batch_size)
    test_generator("../results/test", train_generator)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.MeanSquaredError()

    encoder = get_encoder()
    loss_net = get_style_vgg()
    decoder = get_decoder()

    model = NeuralStyleTransfer(
        encoder=encoder, decoder=decoder, loss_net=loss_net, style_weight=style_weight, alpha=alpha
    )

    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    # mcp_save = ModelCheckpoint('../models/adain/' + now + '.h5', save_best_only=True, monitor='val_total_loss',
    # mode='min')

    reduce_lr = ReduceLROnPlateau(monitor='val_total_loss', factor=0.2,
                                  patience=5, cooldown=15, verbose=1, min_lr=lr_decay)
    # history = model.fit(
    #     train_generator,
    #     epochs=epoch,
    #     validation_data=val_generator,
    #     callbacks=[TrainMonitor(*val_generator.next(), alpha=alpha), reduce_lr],
    # )
    # plot_graphs(history.history)
    now = datetime.now().strftime("%m%d%H:%M")
    # model = model.get_model()
    model = keras.models.load_model("../models/adain/1511w4notblur150e.h5")
    # model.save('../models/adain/' + now + '.h5')

    images = get_gen_images(val_generator, 2)
    style_images = model.predict(images)
    unionTestImages(images[0], style_images, path="../results/style/adain")
