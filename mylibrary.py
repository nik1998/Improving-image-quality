import os
import random
import string
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from sklearn.utils import shuffle
import threading


def cross_validation(data, targets, k=4):
    num_validation_samples = len(data) // k
    data, targets = shuffle(data, targets)
    for fold in range(k):
        validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        training_data = np.concatenate(
            [data[:num_validation_samples * fold], data[num_validation_samples * (fold + 1):]], axis=0)
        validation_targets = targets[num_validation_samples * fold:num_validation_samples * (fold + 1)]
        training_targets = np.concatenate(
            [targets[:num_validation_samples * fold], targets[num_validation_samples * (fold + 1):]], axis=0)
        yield training_data, training_targets, validation_data, validation_targets


def plot_graphs(history):
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.figure()
    acc = history['acc']
    val_acc = history['val_acc']
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def my_augmented_function(image):
    i = random.randint(0, 4)
    im = np.rot90(image, k=i)
    return im


def split_image(image, x):
    small_array = []
    for i in range(x, image.shape[0] + 1, x):
        for j in range(x, image.shape[1] + 1, x):
            small_array.append(image[i - x:i, j - x:j])
    h = image.shape[0] - image.shape[0] % x
    w = image.shape[1] - image.shape[1] % x
    return small_array, h, w


def recursive_read_split(image_path, x, inmemory=True, drop=0.5):
    return np.asarray(prepare_dataset(image_path, x, inmemory, drop))


def prepare_dataset(image_path, x, inmemory=False, drop=0.5):
    print(image_path)
    new_dataset = []
    train_path = 'train_images2/'
    val_path = 'val_images2/'
    test_path = 'test_images2/'
    for dr in os.listdir(image_path):
        abs_path = os.path.join(image_path, dr)
        if os.path.isdir(abs_path):
            new_dataset += prepare_dataset(abs_path, x, inmemory)
        elif 'jpg' == dr[-3:] or 'bmp' == dr[-3:]:
            print('Add file:' + abs_path)
            img = read_image(abs_path)
            ar, _, _ = split_image(img, x)
            if inmemory:
                new_dataset.append(ar)
            else:
                images = shuffle(np.asarray(ar))
                d = len(images)
                images = images[int(drop * d):]
                p = random.random()
                if p < 0.8:
                    save_images(images, train_path)
                elif p < 0.9:
                    save_images(images, val_path)
                else:
                    save_images(images, test_path)
    if inmemory:
        return new_dataset
    return train_path, val_path, test_path


inn = 0


def save_images(images, path, stdNorm=False, imageNames=None):
    global inn
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(images.shape[0]):
        img = images[i]
        if stdNorm:
            img = img * 127.5 + 127.5
        else:
            img = img * 255
        # print(np.mean(img))
        if imageNames is None:
            cv2.imwrite(os.path.join(path, "img" + str(inn) + ".png"), img)
            inn += 1
        else:
            cv2.imwrite(os.path.join(path, imageNames[i]), img)


def debug_get_activations(model, test_image):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    img_tensor = np.expand_dims(test_image, axis=0)
    activations = activation_model.predict(img_tensor)

    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')


def read_image(imageName: string, height=0, width=0):
    im = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    if width != 0 and height != 0:
        im = cv2.resize(im, (height, width))
    return np.asarray(im, dtype=np.float32) / 255


def read_dir(imagePath, height, width, sort=False):
    dir = os.listdir(imagePath)
    if sort:
        dir = sorted(dir)
    dir_images = []
    for l in dir:
        dir_images.append(read_image(os.path.join(imagePath, l), height, width))
    return np.asarray(dir_images)


def showImage(image):
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=0.0, vmax=1.0)
    plt.show()


def unionTestImages(act, predict, hsize=5, wsize=10, path="unionTest/", stdNorm=True):
    conImages = []
    for i in range(0, act.shape[0], wsize * hsize):
        img = None
        for j in range(hsize):
            l = i + wsize * j
            r = i + wsize * j + wsize
            a = act[l:r]
            p = predict[l:r]
            if len(act[l:r]) > 1:
                a = np.hstack(a)
                p = np.hstack(p)
            else:
                if len(a) != 0:
                    a = a[0]
                    p = p[0]
                else:
                    continue
            if img is None:
                img = np.vstack((a, p))
            else:
                img = np.vstack((img, a, p))
        if img is not None:
            conImages.append(img)
    save_images(np.asarray(conImages), path, stdNorm=stdNorm)


def std_norm_x(x):
    return 2 * x - 1


def std_norm_reverse(x):
    return (x + 1) / 2.0


def linear_norm_x(x):
    return x / 255


def brightNowm(all_images):
    m = int(np.mean(all_images))
    for i, im in enumerate(all_images):
        delta = m - int(np.mean(im))
        all_images[i] = np.clip(all_images[i], -delta, 255) + delta
        all_images[i] = np.clip(all_images[i], 0, 255)
        # print(np.min(all_images[i]))
        # print(np.max(all_images[i]))


def plotHist(all_images):
    x = []
    for im in all_images:
        x.append(int(np.mean(im)))
    # the histogram of the data
    plt.hist(x, facecolor='g', alpha=0.75)

    plt.xlabel('Average brightness')
    plt.ylabel('Count of images')
    plt.title('Histogram of brightness')
    # plt.xlim(0, 255)
    plt.grid(True)
    plt.show()


def unionImage(images, h, w):
    if len(images) == 0:
        return None
    sh, sw = images[0].shape
    kh = h // sh
    kw = w // sw
    img = np.hstack(images[0:kw])
    for i in range(1, kh):
        p = np.hstack(images[i * kw:i * kw + kw])
        img = np.vstack((img, p))
    return img


def read_operate_save(image_path, save_path, operate):
    img = read_image(image_path)
    print(image_path)
    sys.stdout.flush()
    img = operate(img)
    cv2.imwrite(save_path, img * 255)


def recursive_read_operate_save(image_path, save_path, operate, timeout=True):
    for dr in os.listdir(image_path):
        abs_path = os.path.join(image_path, dr)
        if os.path.isdir(abs_path):
            recursive_read_operate_save(abs_path, save_path, operate)
        elif 'jpg' == dr[-3:] or 'bmp' == dr[-3:] or 'png' == dr[-3:]:
            s = os.path.join(save_path, dr)
            x = threading.Thread(target=read_operate_save, args=(abs_path, s, operate))
            x.start()
    if timeout:
        time.sleep(1)
