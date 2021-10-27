import os

import keras
import numpy as np
from keras import Sequential
from tensorflow.keras.utils import plot_model
from keras.layers import Conv2D, Conv2DTranspose
from scipy import linalg
from tensorflow.keras.constraints import max_norm
import tensorflow as tf
import keras.backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint
from datetime import datetime

import img_filters
from mylibrary import *
from mykeras_utils import *

# default 2
max_norm_value = 3.0
noise_factor = 1.0
width = 128
height = 128
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_with_cross_val(noisy_data, data_img, recognition_model, k=4):
    k = 4
    history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
    for i in range(20):
        for trainx, trainy, valx, valy in cross_validation(noisy_data, data_img):
            his = recognition_model.fit(trainx, trainy, epochs=1, batch_size=16, validation_data=(valx, valy))
            loss = his.history['loss']
            val_loss = his.history['val_loss']
            acc = his.history['acc']
            val_acc = his.history['val_acc']
            history['loss'].append(loss[0])
            history['val_loss'].append(val_loss[0])
            history['acc'].append(acc[0])
            history['val_acc'].append(val_acc[0])
    plot_graphs(history)
    return recognition_model


# gan loss
def calculate_fid(real_embeddings, generated_embeddings):
    # calculate mean and covariance statistics
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(generated_embeddings, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def pixel_distance(real_images, generated_images):
    d = K.sum(K.abs(real_images - generated_images))
    return 1.0 - d / (K.sum(real_images) + K.sum(generated_images))


if __name__ == '__main__':
    # data_img = read_dir("all_images", height, width, True)
    # data_img = recursive_read_split("want_to_split/", height, False, 0.5)
    # noise_generator = np.random.normal(0, 1, data_img.shape)
    # noisy_data = data_img + noise_factor * noise_generator
    # noisy_data = np.expand_dims(noisy_data, axis=-1)
    gen_model = Sequential()
    gen_model.add(Conv2D(16, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                         kernel_initializer='he_uniform', input_shape=(height, width, 1), padding='same'))
    gen_model.add(
        Conv2D(16, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value), activation='relu',
               kernel_initializer='he_uniform', padding='same'))
    gen_model.add(
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value), activation='relu',
               kernel_initializer='he_uniform', padding='same'))
    gen_model.add(
        Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                        activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    gen_model.add(
        Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), kernel_constraint=max_norm(max_norm_value),
                        activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    gen_model.add(
        Conv2DTranspose(16, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu',
                        kernel_initializer='he_uniform', padding='same'))
    gen_model.add(
        Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))
    gen_model.summary()
    plot_model(gen_model, "enc_dec.png")
    acc = pixel_distance
    acc.__name__ = 'acc'
    gen_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[acc])
    generator = My_Custom_Generator("real/train/", 16, noise_factor)
    nt = len(os.listdir("real/train/train_images/"))
    #for im1, im2 in generator:
        #save_images(np.concatenate((im1, im2), axis=1), "aug_test/")
    validation_generator = My_Custom_Generator("real/val/", 16, noise_factor)
    nv = len(os.listdir("real/val/val_images/"))
    bsize = 16
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    mcp_save = ModelCheckpoint('models/model' + now + '.h5', save_best_only=True, monitor='val_loss', mode='min')
    history = gen_model.fit(generator, epochs=100, batch_size=16, steps_per_epoch=nt // bsize // 2,
                            validation_data=validation_generator, validation_steps=nv // bsize // 4,
                            callbacks=[mcp_save])
    plot_graphs(history.history)
    # img = np.reshape(noisy_data[0], noisy_data[0].shape[:-1])
    # showImage(img)
    # noisy_img = np.asarray([noisy_data[0]])
    # im = recognition_model.predict(noisy_img)
    # showImage(np.reshape(im[0], im[0].shape[:-1]))
    # showImage(data_img[1])
    # correct_img = np.asarray([np.expand_dims(data_img[1], axis=2)])
    # im = recognition_model.predict(correct_img)
    # showImage(np.reshape(im[0], im[0].shape[:-1]))

    # testing
    test = read_dir("real/train/train_images/", height, width)
    test = test[:100]
    plt.figure(1)
    plt.title(1)
    plt.hist((255 * test[9]).ravel(), 256, [0, 255])
    for i, im in enumerate(test):
        test[i] = light_side(im, 5)
            #big_light_hole(im)
        # expansion_algorithm(im, 20, gauss=False)
    plt.figure(2)
    plt.title(2)
    plt.hist((255 * test[9]).ravel(), 256, [0, 255])
    test = np.expand_dims(test, axis=-1)
    p = gen_model.predict(test, batch_size=bsize)
    test = np.reshape(test, test.shape[:-1])
    p = np.reshape(p, p.shape[:-1])
    unionTestImages(test, p, path="finalTest/", stdNorm=False)
    plt.figure(3)
    plt.title(3)
    plt.hist((255 * p[9]).ravel(), 256, [0, 255])
    plt.show()
