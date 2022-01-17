import keras.backend as K
import tqdm
from keras import layers

from utils.mykeras_utils import *

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

width = 128
height = 128
channels = 1


# important articles:
# https://towardsdatascience.com/style-transfer-with-gans-on-hd-images-88e8efcf3716
class AdvLoss(keras.layers.Layer):
    def  __init__(self, **kwargs):
        super(AdvLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori_feature = feature_extractor(x[0])
        gan_feature = feature_extractor(x[1])
        return K.mean(K.square(ori_feature - K.mean(gan_feature, axis=0)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


class CntLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CntLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.abs(ori - gan))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


class EncLoss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(EncLoss, self).__init__(**kwargs)

    def call(self, x, mask=None):
        ori = x[0]
        gan = x[1]
        return K.mean(K.square(g_e(ori) - g_e(gan)))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], 1)


def getBaseModel():
    input_layer = layers.Input(shape=(height, width, channels))

    z = layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', kernel_regularizer='l2')(input_layer)
    z = layers.LeakyReLU()(z)

    z = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2')(z)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU()(z)

    z = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2')(z)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU()(z)

    z = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer='l2')(z)
    z = layers.BatchNormalization()(z)
    z = layers.LeakyReLU()(z)

    model = keras.models.Model(input_layer, z)
    return model


def decoderModel(g_e):
    input_layer = layers.Input(name='input', shape=(height, width, channels))
    x = g_e(input_layer)

    y = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='deconv_1', kernel_regularizer='l2')(x)
    y = layers.LeakyReLU(name='de_leaky_1')(y)

    y = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', name='deconv_2', kernel_regularizer='l2')(y)
    y = layers.LeakyReLU(name='de_leaky_2')(y)

    y = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', name='deconv_3', kernel_regularizer='l2')(y)
    y = layers.LeakyReLU(name='de_leaky_3')(y)

    y = layers.Conv2DTranspose(channels, (1, 1), strides=(1, 1), padding='same', name='decoder_deconv_output',
                               kernel_regularizer='l2', activation='tanh')(y)
    return keras.models.Model(inputs=input_layer, outputs=y)


# ## Generators Encoder
g_e = getBaseModel()
g_e.summary()

# ## Generator

g = decoderModel(g_e)
g.summary()

# ## feature extractor
feature_extractor = getBaseModel()
feature_extractor.summary()

# model for training
input_layer = layers.Input(name='input', shape=(height, width, channels))
gan = g(input_layer)  # g(x)

adv_loss = AdvLoss(name='adv_loss')([input_layer, gan])
cnt_loss = CntLoss(name='cnt_loss')([input_layer, gan])
enc_loss = EncLoss(name='enc_loss')([input_layer, gan])

gan_trainer = keras.models.Model(input_layer, [adv_loss, cnt_loss, enc_loss], name="gan_trainer")


# loss function
def loss(yt, yp):
    return yp


losses = {
    'adv_loss': loss,
    'cnt_loss': loss,
    'enc_loss': loss,
}

lossWeights = {'cnt_loss': 20.0, 'adv_loss': 1.0, 'enc_loss': 1.0}

# compile
gan_trainer.compile(optimizer='adam', loss=losses, loss_weights=lossWeights)
gan_trainer.summary()
# keras.utils.plot_model(gan_trainer, to_file="gen.png")
# ## discriminator
input_layer = layers.Input(name='input', shape=(height, width, channels))

f = feature_extractor(input_layer)

d = layers.GlobalAveragePooling2D(name='glb_avg')(f)
d = layers.Dense(1, activation='sigmoid', name='d_out')(d)

d = keras.models.Model(input_layer, d, name="discriminator")
d.summary()
# keras.utils.plot_model(d, to_file="dis.png")
d.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')

if __name__ == '__main__':
    bz = 16
    seed = random.randint(0, 2 ** 30)
    aug = AugmentationUtils().rescale().zoom_range().horizontal_flip().vertical_flip().brightness_range() \
        .std_norm_after_rescale()
    x_train = aug.create_generator("../datasets/real/train/", batch_size=bz,
                                   target_size=(height, width), seed=seed)
    x_test = aug.create_generator("../datasets/real/val/", batch_size=bz,
                                  target_size=(height, width))
    x_test = get_gen_images(x_test)

    aug = AugmentationUtils().rescale().zoom_range().horizontal_flip().vertical_flip().brightness_range() \
        .add_gauss_noise(var=0.05) \
        .std_norm_after_rescale()
    noise_gen = aug.create_generator("../datasets/real/train/",
                                     batch_size=bz,
                                     target_size=(height, width), seed=seed)
    training_generator = UnionGenerator([x_train, noise_gen], batch_size=bz).reflect_rotate()

    test_generator("../results/test/", training_generator, stdNorm=True)
    # train_data_generator = train_datagen.flow(x=x_train, save_to_dir="aug_dataset2/", save_prefix='aug',
    # save_format='png', batch_size=bz)
    minsu = 10000
    niter = 10000
    for i in tqdm.tqdm(range(niter)):
        ### get batch x, y ###
        x, y = training_generator.next()
        ### train disciminator ###
        d.trainable = True
        gan_trainer.trainable = False
        fake_x = g.predict(y)
        d_x = np.concatenate([x, fake_x], axis=0)
        d_y = np.concatenate([np.ones(len(fake_x)), np.zeros(len(fake_x))], axis=0)
        d_loss = d.train_on_batch(d_x, d_y)
        ### train generator ###
        d.trainable = False
        gan_trainer.trainable = True
        g_loss = gan_trainer.train_on_batch(x, np.asarray(y))

        if i % 50 == 0:
            print(f'iter: {i}, g_loss: {g_loss}, d_loss: {d_loss}')
            su = g_loss[0] + d_loss[0]
            if su < minsu:
                minsu = su
                print("save best")
                g_e.save_weights("../models/my_gan/g_e.h5")
                g.save_weights("../models/my_gan/g.h5")
                d.save_weights("../models/my_gan/d.h5")
                g_e.save('../models/my_gan/generator_encoder.h5')
                g.save('../models/my_gan/generator.h5')
                d.save('../models/my_gan/discriminator.h5')
                print("Validation: " + str(d.evaluate(x_test, np.ones(len(x_test)))))
    g_e.load_weights("../models/my_gan/g_e.h5")
    g.load_weights("../models/my_gan/g.h5")
    d.load_weights("../models/my_gan/d.h5")
    # # Evaluation
    encoded = g_e.predict(x_test)
    gan_x = g.predict(x_test)
    encoded_gan = g_e.predict(gan_x)
    score = np.sum(np.absolute(encoded - encoded_gan), axis=-1)
    score = (score - np.min(score)) / (np.max(score) - np.min(score))  # map to 0~1
    print("Accuracy: " + str(d.evaluate(x_test, np.ones(len(x_test)))))

    i = 4  # or 1
    image = np.reshape(gan_x[i:i + 1], (height, width))
    showImage(std_norm_reverse(image))
    image = np.reshape(x_test[i:i + 1], (height, width))
    showImage(std_norm_reverse(image))

    gan_x = np.reshape(gan_x, gan_x.shape[:-1])
    x_test = np.reshape(x_test, x_test.shape[:-1])
    unionTestImages(x_test, gan_x, 1, 1, "../results/images_test_union/", True)

    x_train = get_gen_images(x_train)
    gan_x = g.predict(x_train)
    gan_x = np.reshape(gan_x, gan_x.shape[:-1])
    x_train = np.reshape(x_train, x_train.shape[:-1])
    unionTestImages(x_train, gan_x, 1, 1, "../results/images_train_union/", True)
