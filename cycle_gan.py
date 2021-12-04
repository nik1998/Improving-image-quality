import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
import tensorflow_addons as tfa
from mylibrary import *
from sklearn.utils import shuffle

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

height, width = 128, 128
img_channel = 1
# Define the standard image size.
orig_img_size = (height, width)
# Size of the random crops to be used during training.
input_img_size = (height, width, img_channel)
full_size = height * width * img_channel
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

batch_size = 1


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


datasetB = read_dir("image_for_cycle", height, width, True)
datasetB = shuffle(datasetB)

for i in range(1400):
    datasetB[i] = cv2.medianBlur(datasetB[i], 5)
    datasetB[i] = cv2.GaussianBlur(datasetB[i], (0, 0), 1)

datasetB = 2 * np.expand_dims(datasetB, axis=-1) - 1
for i in range(1400):
    datasetB[i] = tf.image.random_flip_left_right(datasetB[i])

datasetA = read_dir("final_bin_2000", height, width, True)
datasetA = shuffle(datasetA)
datasetA = 2 * np.expand_dims(datasetA, axis=-1) - 1

for i in range(1400):
    datasetA[i] = tf.image.random_flip_left_right(datasetA[i])

# dataset = tf.convert_to_tensor(dataset)
train_A, train_B = datasetA[0:2000], datasetB[0:2000]
test_A, test_B = datasetA[2000:2100], datasetB[2000:2100]
train_A = tf.data.Dataset.from_tensor_slices(train_A).batch(batch_size)
train_B = tf.data.Dataset.from_tensor_slices(train_B).batch(batch_size)
test_A = tf.data.Dataset.from_tensor_slices(test_A).batch(batch_size)
test_B = tf.data.Dataset.from_tensor_slices(test_B).batch(batch_size)
"""
## Visualize some samples
"""

_, ax = plt.subplots(4, 2, figsize=(10, 15))
for i, samples in enumerate(zip(train_A.take(4), train_B.take(4))):
    A = (((samples[0][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    B = (((samples[1][0] * 127.5) + 127.5).numpy()).astype(np.uint8)
    ax[i, 0].imshow(A, 'gray')
    ax[i, 1].imshow(B, 'gray')
plt.show()
"""
## Building blocks used in the CycleGAN generators and discriminators
"""


class ReflectionPadding2D(layers.Layer):
    """Implements Reflection Padding as a layer.
    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.
    Returns:
        A padded tensor with the same type as the input tensor.
    """

    def __init__(self, padding=(1, 1), name=None, **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        return config


def residual_block(x, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                   gamma_initializer=gamma_init, use_bias=False,
                   ):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding,
                      use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding,
                      use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


def downsample(
        x,
        filters,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        gamma_initializer=gamma_init,
        use_bias=False,
):
    x = layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding,
                      use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_init,
        gamma_initializer=gamma_init,
        use_bias=False,
):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
                               kernel_initializer=kernel_initializer, use_bias=use_bias)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_resnet_generator(filters=64, num_downsampling_blocks=2, num_residual_blocks=9, num_upsample_blocks=2,
                         gamma_initializer=gamma_init, name=None, ):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation("relu"))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(img_channel, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


"""
## Build the discriminators
The discriminators implement the following architecture:
`C64->C128->C256->C512`
"""


def get_discriminator(filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None):
    img_input = layers.Input(shape=input_img_size, name=name + "_img_input")
    x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2))
        else:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1))

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)
    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


# Get the generators
gen_G = get_resnet_generator(name="generator_G")
gen_F = get_resnet_generator(name="generator_F")

gen_G.summary()
gen_F.summary()
# Get the discriminators
disc_X = get_discriminator(name="discriminator_X")
disc_Y = get_discriminator(name="discriminator_Y")
disc_X.summary()
disc_Y.summary()
gen_G.summary()
gen_F.summary()


def pixel_distance(real_images, generated_images):
    d = K.sum(K.square(real_images - generated_images))
    return d / full_size


class CycleGan(keras.Model):
    def __init__(self, generator_G, generator_F, discriminator_X, discriminator_Y, lambda_cycle=10.0,
                 lambda_identity=0.5):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer, gen_loss_fn, disc_loss_fn):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    # override
    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
            )
            id_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        acc = (pixel_distance(real_x, cycled_x) + pixel_distance(real_y, cycled_y)) / 2
        return {
            "acc": acc,
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss,
        }

    def save(self, name, **kwargs):
        self.gen_G.save(name + "gen_G.h5")
        self.gen_F.save(name + "gen_F.h5")
        self.disc_X.save(name + "disc_X.h5")
        self.disc_Y.save(name + "disc_Y.h5")

    def save_weights(self, name, **kwargs):
        if not os.path.exists(name):
            os.makedirs(name)
        self.gen_G.save_weights(name + "gen_G.h5")
        self.gen_F.save_weights(name + "gen_F.h5")
        self.disc_X.save_weights(name + "disc_X.h5")
        self.disc_Y.save_weights(name + "disc_Y.h5")

    def load_weights(self, name, **kwargs):
        self.gen_G.load_weights(name + "gen_G.h5")
        self.gen_F.load_weights(name + "gen_F.h5")
        self.disc_X.load_weights(name + "disc_X.h5")
        self.disc_Y.load_weights(name + "disc_Y.h5")


"""
## Create a callback that periodically saves generated images
"""


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test_A.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img, 'gray')
            ax[i, 1].imshow(prediction, 'gray')
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.preprocessing.image.array_to_img(prediction)
            prediction.save("cycle_result/generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1))
        plt.show()
        plt.close()


"""
## Train the end-to-end model
"""

# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()


# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)


def train(weight_file=""):
    if weight_file != "":
        cycle_gan_model.load_weights(weight_file)
    # Callbacks
    plotter = GANMonitor()
    # checkpoint_filepath = "./model_checkpoints/cyclegan_checkpoints.{epoch:03d}"
    now = datetime.now().strftime("%m/%d/%H:%M/")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint('cycle_result/models/model' + now, save_best_only=True,
                                                                monitor='acc', mode='min', save_weights_only=True)
    # model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath)

    history = cycle_gan_model.fit(tf.data.Dataset.zip((train_A, train_B)), epochs=3,
                                  callbacks=[plotter, model_checkpoint_callback])
    plot_graphs(history.history)
    # cycle_gan_model.save(name='models/model' + now)


def plot_graphs(history):
    for key in history:
        plt.figure()
        loss = history[key]
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss)
        plt.title(key)
        plt.xlabel('Epochs')
        plt.ylabel(key)
    plt.show()


def test():
    # Load the checkpoints
    weight_file = "cycle_result/models/model11/17/16:08/"
    cycle_gan_model.load_weights(weight_file)

    _, ax = plt.subplots(4, 2, figsize=(10, 15))
    for i, img in enumerate(test_A.take(4)):
        prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        ax[i, 0].imshow(img, 'gray')
        ax[i, 1].imshow(prediction, 'gray')
        ax[i, 0].set_title("Input image")
        ax[i, 0].set_title("Input image")
        ax[i, 1].set_title("Translated image")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")

        prediction = keras.preprocessing.image.array_to_img(prediction)
        prediction.save("cycle_result/predicted_img_{i}.png".format(i=i))
    plt.tight_layout()
    plt.show()


def check():
    weight_file = "cycle_result/models/model11/17/18:52/"
    cycle_gan_model.load_weights(weight_file)
    res = []
    for i, img in enumerate(test_A):
        prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
        prediction = prediction
        img = img[0].numpy()
        res.append(np.concatenate((img, prediction), axis=1))
    save_images(np.asarray(res), 'cycle_result/test_images3/', stdNorm=True)


def check2():
    weight_file = "cycle_result/models/model11/17/16:08/"
    cycle_gan_model.load_weights(weight_file)
    res = []
    for i, img in enumerate(test_B):
        prediction = cycle_gan_model.gen_F(img, training=False)[0].numpy()
        prediction = prediction
        img = img[0].numpy()
        res.append(np.concatenate((img, prediction), axis=1))
    save_images(np.asarray(res), 'cycle_result/test_images2/', stdNorm=True)


mode_test = True
if __name__ == '__main__':
    if mode_test:
        check()
        # test()
    else:
        train()
