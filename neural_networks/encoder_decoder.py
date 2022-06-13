from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm

from utils.mykeras_utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def pixel_distance(real_images, generated_images):
    d = K.sum(K.abs(real_images - generated_images))
    return 1.0 - d / (K.sum(real_images) + K.sum(generated_images))


def noise_function(aug: AugmentationUtils):
    return aug.add_different_noise(p=0.8)


def create_enc_dec():
    width = 256
    height = 256
    max_norm_value = 2.0
    repair_model = Sequential(
        [
            Conv2D(16, kernel_size=(5, 5), kernel_constraint=max_norm(max_norm_value), activation='relu',
                   kernel_initializer='he_uniform', input_shape=(height, width, 1), padding='same'),

            Conv2D(16, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                   kernel_initializer='he_uniform', input_shape=(height, width, 1), padding='same'),

            Conv2D(32, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                   activation='relu',
                   kernel_initializer='he_uniform', padding='same'),

            Conv2D(64, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                   activation='relu',
                   kernel_initializer='he_uniform', padding='same'),
            Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
            Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                            activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
            Conv2DTranspose(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                            kernel_initializer='he_uniform', padding='same'),
            Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid',
                   padding='same')
        ])
    loss_fn = keras.losses.MeanSquaredError()
    repair_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    return repair_model


if __name__ == '__main__':
    batch_size = 32
    image_dir = '../datasets/not_sem/cats/real'
    im_size = 256

    # repair_model.summary()
    # plot_model(repair_model, "enc_dec.png")
    repair_model = create_enc_dec()

    train_generator, val_generator = create_image_to_image_generator([image_dir, image_dir],
                                                                     aug_extension=[noise_function],
                                                                     batch_size=batch_size,
                                                                     im_size=im_size, vertical_flip=False,
                                                                     ninty_rotate=False)

    callbacks = get_default_callbacks("../models/enc_cats", train_generator, repair_model)

    history = repair_model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=callbacks)

    plot_graphs(history.history)

    images = get_gen_images(val_generator, 100)

    repair_model.load_weights(get_latest_filename("../models/enc_cats"))
    test = repair_model.predict(images[0])

    unionTestImages(images[0], test, path="../results/enc_dec_cats")
    save_images(np.concatenate([images[0], test], axis=2), path="../results/enc_dec_cats")
