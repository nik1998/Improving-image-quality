import os
import random
import shutil
import string
import sys
import threading
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def wrap(img, add_channels=False):
    if add_channels:
        img = np.expand_dims(img, axis=-1)
    return np.asarray([img])


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
    train_path = '../datasets/train_images/'
    val_path = 'val_images/'
    test_path = '../datasets/test_images/'
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
    for i in range(len(images)):
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


def read_image(imageName: string, height=0, width=0, gray=True):
    if gray:
        im = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.imread(imageName)
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
            # read_operate_save(abs_path, s, operate)
            x = threading.Thread(target=read_operate_save, args=(abs_path, s, operate))
            x.start()
    if timeout:
        time.sleep(1)


def copy_from_labels(labels_path, source_path, dst_path):
    dir = os.listdir(labels_path)
    for name in dir:
        cut_name = name.split("/")[-1]
        src_img = os.path.join(source_path, cut_name)
        dst_img = os.path.join(dst_path, cut_name)
        shutil.copyfile(src_img, dst_img)


def check_model(model, post_func):
    def f(img):
        im2 = np.reshape(img, (1,) + img.shape + (1,))
        pred = model.predict(im2, batch_size=1)
        p = post_func(pred[0])
        p = np.reshape(p, p.shape[:-1])
        return np.concatenate([img, p])

    return f


def simple_boundary(img):
    return np.around(img)


def check_not_interception(arr1, arr2):
    s = set(arr1)
    for e in arr2:
        if e in s:
            return False
    return True
