from datetime import datetime

import keras.backend as K
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.constraints import max_norm
from tensorflow.python.keras.callbacks import ModelCheckpoint

from utils.mykeras_utils import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def pixel_distance(real_images, generated_images):
    d = K.sum(K.abs(real_images - generated_images))
    return 1.0 - d / (K.sum(real_images) + K.sum(generated_images))


if __name__ == '__main__':
    max_norm_value = 2.0
    width = 128
    height = 128
    seed = random.randint(0, 2 ** 30)
    batch_size = 16

    repair_model = Sequential(
        [
            Conv2D(16, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                   kernel_initializer='he_uniform', input_shape=(height, width, 1), padding='same'),

            Conv2D(16, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                   activation='relu',
                   kernel_initializer='he_uniform', padding='same'),

            Conv2D(32, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                   activation='relu',
                   kernel_initializer='he_uniform', padding='same'),
            Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
            Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
            Conv2DTranspose(16, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
            Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid',
                   padding='same')
        ])
    # repair_model.summary()
    # plot_model(repair_model, "enc_dec.png")
    acc = pixel_distance
    acc.__name__ = 'acc'
    repair_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[acc])

    aug = AugmentationUtils(). \
        rescale(). \
        horizontal_flip(). \
        vertical_flip(). \
        brightness_range(). \
        zoom_range()
    train_generator = aug.create_generator("../datasets/real/train/", seed=seed)
    nt = train_generator.samples

    validation_generator = aug.create_generator("../datasets/real/val/", seed=seed)
    nv = validation_generator.samples

    aug = aug.add_gauss_noise(var=0.02).add_big_light_hole().add_light_side().add_big_own_defect().\
        add_defect_expansion_algorithm().add_salt_paper()

    train_generator2 = aug.create_generator("../datasets/real/train/", seed=seed)
    train_generator = UnionGenerator([train_generator, train_generator2], batch_size=batch_size)
    validation_generator2 = aug.create_generator("../datasets/real/val/", seed=seed)
    validation_generator = UnionGenerator([validation_generator, validation_generator2],
                                          batch_size=batch_size)
    test_generator("../results/test", train_generator)
    test_generator("../results/test", validation_generator)

    now = datetime.now().strftime("%m%d%H:%M")

    mcp_save = ModelCheckpoint('../models/enc_dec/model' + now + '.h5', save_best_only=True, monitor='val_loss',
                               mode='min')
    history = repair_model.fit(train_generator, epochs=100, batch_size=batch_size, steps_per_epoch=nt // batch_size,
                               validation_data=validation_generator, validation_steps=nv // batch_size,
                               callbacks=[mcp_save])
    plot_graphs(history.history)

    # testing
    test = read_dir("../datasets/real/train/train_images/", height, width)
    test = test[:100]
    plt.figure(1)
    plt.title(1)
    plt.hist((255 * test[9]).ravel(), 256, [0, 255])
    # for i, im in enumerate(test):
    # test[i] = light_side(im, 5)
    # big_light_hole(im)
    # expansion_algorithm(im, 20, gauss=False)
    # plt.figure(2)
    # plt.title(2)
    # plt.hist((255 * test[9]).ravel(), 256, [0, 255])
    test = np.expand_dims(test, axis=-1)
    p = repair_model.predict(test, batch_size=batch_size)
    test = np.reshape(test, test.shape[:-1])
    p = np.reshape(p, p.shape[:-1])
    unionTestImages(test, p, path="../results/enc_dec/results/", stdNorm=False)
    plt.figure(3)
    plt.title(3)
    plt.hist((255 * p[9]).ravel(), 256, [0, 255])
    plt.show()
