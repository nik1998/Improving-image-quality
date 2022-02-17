from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from utils.mykeras_utils import *


def get_encoder(input_img_size):
    vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=input_img_size)
    vgg19.trainable = False
    mini_vgg19 = keras.Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    inputs = layers.Input(input_img_size)
    mini_vgg19_out = mini_vgg19(inputs)
    return keras.Model(inputs, mini_vgg19_out, name="mini_vgg19")


def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]
    mean, variance = tf.nn.moments(x, axes=axes, keepdims=True)
    standard_deviation = tf.sqrt(variance + epsilon)
    return mean, standard_deviation


def ada_in(style, content, alpha=1.0, batch=None):
    if batch is not None:
        return batch(content)
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


def get_style_vgg(input_img_size):
    vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=input_img_size)
    vgg19.trainable = False
    layer_names = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"]
    outputs = [vgg19.get_layer(name).output for name in layer_names]
    mini_vgg19 = keras.Model(vgg19.input, outputs)

    inputs = layers.Input(input_img_size)
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
        self.loss_fn = None
        self.optimizer = None
        self.encoder = encoder
        self.decoder = decoder
        self.loss_net = loss_net
        self.style_weight = style_weight
        self.alpha = alpha
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()
        self.batch = layers.BatchNormalization()

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def call(self, input_tensor, mask=None, **kwargs):
        pass

    def predict(self, input_tensor):
        style, content = input_tensor
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        t = ada_in(style=style_encoded, content=content_encoded, alpha=self.alpha, batch=self.batch)
        reconstructed_image = self.decoder(t)
        return reconstructed_image

    def get_losses(self, inputs):
        style, content = inputs

        # Encode the style and content image.
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN target feature maps.
        t = ada_in(style=style_encoded, content=content_encoded, alpha=self.alpha, batch=self.batch)

        # Generate the neural style transferred image.
        reconstructed_image = self.decoder(t)
        # Compute the losses.
        reconstructed_vgg_features = self.loss_net(reconstructed_image)
        style_vgg_features = self.loss_net(style)
        content_loss = self.loss_fn(t, reconstructed_vgg_features[-1])
        style_loss = 0.0
        for inp, out in zip(style_vgg_features, reconstructed_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            style_loss += self.loss_fn(mean_inp, mean_out) + self.loss_fn(std_inp, std_out)
        style_loss = self.style_weight * style_loss
        identity_loss = self.identity_loss_fn(self.decoder(content_encoded), content) * 100
        total_loss = content_loss + style_loss + identity_loss
        return {
            "style_loss": style_loss,
            "content_loss": content_loss,
            "identity_loss": identity_loss,
            "total_loss": total_loss,
        }

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            losses = self.get_losses(inputs)
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(losses["total_loss"], trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return losses

    def test_step(self, inputs):
        return self.get_losses(inputs)

    def get_model(self, input_img_size):
        inputs1 = layers.Input(input_img_size)
        inputs2 = layers.Input(input_img_size)
        input1 = self.encoder(inputs1)
        input2 = self.encoder(inputs2)
        t = ada_in(style=input1, content=input2, alpha=self.alpha, batch=self.batch)
        x = self.decoder(t)
        return keras.Model([inputs1, inputs2], x)


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
    lr_decay = 5e-5
    im_size = 256
    batch_size = 8
    style_weight = 0
    initial_img_size = (im_size, im_size, 3)
    alpha = 0.0
    epoch = 200

    style_dir = "../datasets/style"
    content_dir = '../datasets/unet/small_scale/all_real_images'

    train_generator, val_generator = create_image_to_image_generator([style_dir, content_dir],
                                                                     batch_size=batch_size,
                                                                     im_size=im_size,
                                                                     color_mode='rgb',
                                                                     different_seed=True)

    test_generator("../results/test", train_generator, 200)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    loss_fn = keras.losses.MeanSquaredError()

    encoder = get_encoder(initial_img_size)
    get_encoder(initial_img_size).summary()
    loss_net = get_style_vgg(initial_img_size)
    decoder = get_decoder()

    model = NeuralStyleTransfer(
        encoder=encoder, decoder=decoder, loss_net=loss_net, style_weight=style_weight, alpha=alpha
    )

    model.compile(optimizer=optimizer, loss_fn=loss_fn)

    callbacks = get_callbacks('../models/adain/', train_pred_func=style_image_predict(val_generator, model),
                              monitor_loss='val_total_loss')

    callbacks.append(ReduceLROnPlateau(monitor='val_total_loss', factor=0.2,
                                       patience=5, cooldown=15, verbose=1, min_lr=lr_decay))
    history = model.fit(train_generator, epochs=epoch, validation_data=val_generator, callbacks=callbacks)
    plot_graphs(history.history)

    images = get_gen_images(val_generator)
    stylized_images = model.predict(images)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    images = images[0]
    images = np.dot(images[..., :3], rgb_weights)
    stylized_images = np.dot(stylized_images[..., :3], rgb_weights)
    unionTestImages(images, stylized_images, path="../results/style/adain")


if __name__ == "__main__":
    train()
