import keras.backend as K
import tensorflow_addons as tfa

from utils.mykeras_utils import *
from utils.presision_recall_gan import knn_precision_recall_features

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

im_size = 256
img_channel = 1
# Define the standard image size.
orig_img_size = (im_size, im_size)
# Size of the random crops to be used during training.
input_img_size = (im_size, im_size, img_channel)
full_size = im_size * im_size * img_channel
# Weights initializer for the layers.
kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
# Gamma initializer for instance normalization.
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
batch_size = 2
# Loss function for evaluating adversarial loss
adv_loss_fn = keras.losses.MeanSquaredError()

ones = tf.ones(batch_size)
zeros = tf.zeros(batch_size)

imageA_path = "../datasets/cycle/sem_to_sem256/imgsA"
imageB_path = "../datasets/cycle/sem_to_sem256/imgsB"


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
        strides=(1, 1),
        padding="same",
        kernel_initializer=kernel_init,
        gamma_initializer=gamma_init,
        use_bias=False,
):
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer, use_bias=use_bias)(x)
    # x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding,
    # kernel_initializer=kernel_initializer, use_bias=use_bias)(x)
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


def pixel_distance(real_images, generated_images):
    d = K.sum(K.square(real_images - generated_images))
    return d / full_size


class CycleGan(keras.Model):
    def __init__(self, generator_AB, generator_BA, discriminator_B, discriminator_A, lambda_cycle=10.0,
                 lambda_identity=0.5):
        super(CycleGan, self).__init__()
        self.gen_AB = generator_AB
        self.gen_BA = generator_BA
        self.disc_B = discriminator_B
        self.disc_A = discriminator_A
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, gen_AB_optimizer, gen_BA_optimizer, disc_B_optimizer, disc_A_optimizer, gen_loss_fn,
                disc_loss_fn):
        super(CycleGan, self).compile()
        self.gen_AB_optimizer = gen_AB_optimizer
        self.gen_BA_optimizer = gen_BA_optimizer
        self.disc_B_optimizer = disc_B_optimizer
        self.disc_A_optimizer = disc_A_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def call(self, input_tensor, mask=None, **kwargs):
        pass

    def _get_loss(self, real_a, real_b):
        fake_b = self.gen_AB(real_a, training=True)
        fake_a = self.gen_BA(real_b, training=True)

        cycled_a = self.gen_BA(fake_b, training=True)
        cycled_b = self.gen_AB(fake_a, training=True)

        # Identity mapping
        # same_a = self.gen_BA(real_a, training=True)
        # same_b = self.gen_AB(real_b, training=True)

        # Discriminator output
        disc_real_a = self.disc_A(real_a, training=True)
        disc_fake_a = self.disc_A(fake_a, training=True)

        disc_real_b = self.disc_B(real_b, training=True)
        disc_fake_b = self.disc_B(fake_b, training=True)

        # Generator adverserial loss
        gen_AB_loss = self.generator_loss_fn(disc_fake_b)
        gen_BA_loss = self.generator_loss_fn(disc_fake_a)

        # Generator cycle loss
        cycle_loss_B = self.cycle_loss_fn(real_b, cycled_b) * self.lambda_cycle
        cycle_loss_A = self.cycle_loss_fn(real_a, cycled_a) * self.lambda_cycle

        # Generator identity loss
        # id_loss_B = self.identity_loss_fn(real_b, same_b) * self.lambda_identity
        # id_loss_A = self.identity_loss_fn(real_a, same_a) * self.lambda_identity

        # Total generator loss
        total_loss_AB = gen_AB_loss + cycle_loss_B + cycle_loss_A  # + id_loss_B
        total_loss_BA = gen_BA_loss + cycle_loss_A + cycle_loss_B  # + id_loss_A

        # Discriminator loss
        disc_A_loss = self.discriminator_loss_fn(disc_real_a,
                                                 disc_fake_a)
        disc_B_loss = self.discriminator_loss_fn(disc_real_b,
                                                 disc_fake_b)
        return total_loss_AB, total_loss_BA, disc_B_loss, disc_A_loss

    # override
    def train_step(self, batch_data):
        real_a, real_b = batch_data

        with tf.GradientTape(persistent=True) as tape:
            total_loss_AB, total_loss_BA, disc_B_loss, disc_A_loss = self._get_loss(real_a, real_b)
        # Get the gradients for the generators
        grads_AB = tape.gradient(total_loss_AB, self.gen_AB.trainable_variables)
        grads_BA = tape.gradient(total_loss_BA, self.gen_BA.trainable_variables)

        # Get the gradients for the discriminators
        disc_B_grads = tape.gradient(disc_B_loss, self.disc_B.trainable_variables)
        disc_A_grads = tape.gradient(disc_A_loss, self.disc_A.trainable_variables)

        # Update the weights of the generators
        self.gen_AB_optimizer.apply_gradients(
            zip(grads_AB, self.gen_AB.trainable_variables)
        )
        self.gen_BA_optimizer.apply_gradients(
            zip(grads_BA, self.gen_BA.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_B_optimizer.apply_gradients(
            zip(disc_B_grads, self.disc_B.trainable_variables)
        )
        self.disc_A_optimizer.apply_gradients(
            zip(disc_A_grads, self.disc_A.trainable_variables)
        )

        # acc = (pixel_distance(real_a, cycled_a) + pixel_distance(real_b, cycled_b)) / 2
        return {
            "AB_loss": total_loss_AB,
            "BA_loss": total_loss_BA,
            "D_A_loss": disc_A_loss,
            "D_B_loss": disc_B_loss,
        }

        # override

    def test_step(self, batch_data):
        real_a, real_b = batch_data
        total_loss_AB, total_loss_BA, disc_B_loss, disc_A_loss = self._get_loss(real_a, real_b)

        return {
            "G_loss": total_loss_AB + total_loss_BA,
            "D_loss": disc_A_loss + disc_B_loss,
        }

    def save(self, name, **kwargs):
        self.gen_AB.save(name + "/gen_AB.h5")
        self.gen_BA.save(name + "/gen_BA.h5")
        self.disc_B.save(name + "/disc_B.h5")
        self.disc_A.save(name + "/disc_A.h5")

    def save_weights(self, name, **kwargs):
        if not os.path.exists(name):
            os.makedirs(name)
        self.gen_AB.save_weights(name + "/gen_AB.h5")
        self.gen_BA.save_weights(name + "/gen_BA.h5")
        self.disc_B.save_weights(name + "/disc_B.h5")
        self.disc_A.save_weights(name + "/disc_A.h5")

    def load_weights(self, name, **kwargs):
        self.gen_AB.load_weights(name + "/gen_AB.h5")
        self.gen_BA.load_weights(name + "/gen_BA.h5")
        self.disc_B.load_weights(name + "/disc_B.h5")
        self.disc_A.load_weights(name + "/disc_A.h5")


def cycle_image_to_image_func(gen, model):
    def f():
        t_a, t_b = gen.next()
        test_A = t_a[:2]
        test_B = t_b[:2]
        prediction_B = model.gen_AB.predict(test_A)
        prediction_A = model.gen_BA.predict(test_B)
        prediction = np.concatenate([prediction_B, prediction_A], 0)
        train_x = np.concatenate([test_A, test_B], 0)

        return train_x, prediction

    return f


def dataset_function(aug: AugmentationUtils):
    # add noise +4
    return aug.ninty_rotation().zoom_range(l=0.7, r=1).add_gauss_noise(0, 0.00025)


# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


def wasserstein_loss(real, fake):
    return (tf.reduce_sum(real * ones) + tf.reduce_sum(fake * zeros)) / batch_size / 2


def get_cycleGAN():
    # Get the generators
    gen_AB = get_resnet_generator(name="generator_AB")
    gen_BA = get_resnet_generator(name="generator_BA")
    # Get the discriminators
    disc_B = get_discriminator(name="discriminator_B")
    disc_A = get_discriminator(name="discriminator_A")
    # disc_B.summary()
    # disc_A.summary()
    # gen_AB.summary()
    # gen_BA.summary()

    # Create cycle gan model
    cycle_gan_model = CycleGan(
        generator_AB=gen_AB, generator_BA=gen_BA, discriminator_B=disc_B, discriminator_A=disc_A,
        lambda_cycle=10,
        lambda_identity=0.1
    )

    # Compile the model
    cycle_gan_model.compile(
        gen_AB_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_BA_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_A_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_B_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn,
    )
    return cycle_gan_model


def train(cycle_gan_model, train_dataset, test_dataset, weight_file=""):
    if weight_file != "":
        cycle_gan_model.load_weights(weight_file)
    # Callbacks
    callbacks = get_callbacks('../models/cycleGAN', cycle_image_to_image_func(test_dataset, cycle_gan_model),
                              monitor_loss='val_G_loss')

    now = datetime.now().strftime("%m%d%H:%M")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint('../models/cycleGAN/model{epoch}' + now,
                                                                verbose=1,
                                                                save_freq=10 * len(train_dataset),
                                                                save_weights_only=True)
    callbacks.append(model_checkpoint_callback)
    t = len(train_dataset)
    v = len(test_dataset)

    history = cycle_gan_model.fit(train_dataset, steps_per_epoch=t, epochs=10,
                                  validation_data=test_dataset,
                                  validation_steps=v,
                                  callbacks=callbacks)
    plot_graphs(history.history)


def test(cycle_gan_model, test_A):
    _, ax = plt.subplots(4, 2, figsize=(12, 12))
    imgs = get_gen_images(test_A, count=4)
    predictions = cycle_gan_model.gen_AB(imgs, training=False)
    predictions = (predictions * 127.5 + 127.5).astype(np.uint8)
    imgs = (imgs * 127.5 + 127.5).astype(np.uint8)
    for i, img, prediction in enumerate(zip(imgs, predictions)):
        ax[i, 0].imshow(img, 'gray')
        ax[i, 1].imshow(prediction, 'gray')
        ax[i, 0].set_title("Input image")
        ax[i, 0].set_title("Input image")
        ax[i, 1].set_title("Translated image")
        ax[i, 0].axis("off")
        ax[i, 1].axis("off")
        prediction = keras.preprocessing.image.array_to_img(prediction)
        prediction.save("../results/cycle_result/predicted_img_{i}.png".format(i=i))
    plt.tight_layout()
    plt.show()


def check(cycle_gan_model, test_A):
    test_A.seed = 0
    imgs = get_gen_images(test_A)
    test_A.seed = 0
    predictions = cycle_gan_model.gen_AB.predict(test_A)
    # unionTestImages(imgs, predictions, path='../results/cycle_result/test_images/', stdNorm=True)
    save_images(np.hstack([imgs, predictions]), path='../results/cycle_result/test_images/', stdNorm=True)


def check2(cycle_gan_model, test_B):
    test_B.seed = 0
    imgs = get_gen_images(test_B)
    test_B.seed = 0
    predictions = cycle_gan_model.gen_BA.predict(test_B)
    # unionTestImages(imgs, predictions, path='../results/cycle_result/test_images2/', stdNorm=True)
    save_images(np.hstack([imgs, predictions]), path='../results/cycle_result/test_images2/', stdNorm=True)


def generate_for_layer(cycle_gan_model, image_path, atob=True):
    image = std_norm_x(read_image(image_path))
    images, _, _ = split_image(image, im_size, step=128)
    images = np.asarray(images)
    if atob:
        predictions = cycle_gan_model.gen_AB.predict(images, batch_size=batch_size)
    else:
        predictions = cycle_gan_model.gen_BA.predict(images, batch_size=batch_size)
    predictions = np.asarray(predictions)
    predictions = std_norm_reverse(np.reshape(predictions, predictions.shape[:-1]))
    concat_clip_save(predictions, "../results/cycle_result/test_images3/im.png", int(math.sqrt(predictions.shape[0])))


mode_test = True
if __name__ == '__main__':
    model = get_cycleGAN()
    train_generator, val_generator = create_image_to_image_dataset([imageA_path, imageB_path],
                                                                   batch_size=batch_size,
                                                                   im_size=im_size,
                                                                   different_seed=True, stdNorm=True)
    # test_generator("../results/test", train_A, stdNorm=True)
    # test_generator("../results/test", UnionGenerator([val_A, val_B]), stdNorm=True)
    if mode_test:
        weight_file = "../models/cycleGAN/model012514:29"
        model.load_weights(weight_file)
        check2(model, val_generator.generators[1])
        check(model, val_generator.generators[0])
        generate_for_layer(model, "../datasets/cycle/test/cy2_m1_0518.jpg", atob=False)
    else:
        train(model, train_generator, val_generator)
        t_a = get_gen_images(train_generator.generators[0])
        t_b = get_gen_images(train_generator.generators[1])
        prediction_B = model.gen_AB.predict(train_generator.generators[0])
        prediction_A = model.gen_BA.predict(train_generator.generators[1])
        model = None
        knn_precision_recall_features(t_a, prediction_A, row_batch_size=100, col_batch_size=100)
        knn_precision_recall_features(t_b, prediction_B, row_batch_size=100, col_batch_size=100)
