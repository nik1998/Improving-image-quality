from utils.mykeras_utils import *
from tensorflow.keras.callbacks import ModelCheckpoint

from datetime import datetime

if __name__ == '__main__':
    batch_size = 32
    image_dir = "../datasets/final_good_images"
    im_size = 256

    input_img = keras.Input(shape=(im_size, im_size, 1))

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()

    train_generator, val_generator = create_image_to_image_dataset(image_dir, image_dir, batch_size=batch_size,
                                                                   im_size=im_size)
    test_generator("../results/test", train_generator, 100)

    callbacks = get_callbacks("../models/enc_dec_simple", autoencoder, train_generator)

    history = autoencoder.fit(train_generator, epochs=50, batch_size=batch_size, validation_data=val_generator,
                              callbacks=callbacks)

    plot_graphs(history.history)

    images = get_gen_images(val_generator, 100)

    test = autoencoder.predict(images[0])

    unionTestImages(images[0], test, path="..results/enc_dec_simple")
