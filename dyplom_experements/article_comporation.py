import ssl

import numpy as np

from dyplom_experements.mirnet import mirnet_model, charbonnier_loss, peak_signal_noise_ratio

ssl._create_default_https_context = ssl._create_unverified_context
import os.path

from tensorflow import keras

from neural_networks.REDnet import REDNet
from utils.image_quality import mean_psnr, mean_ssim

from utils.mykeras_utils import get_gen_images, create_image_to_image_generator, get_fid, get_default_callbacks
from utils.mylibrary import get_latest_filename, plot_graphs, unionTestImages, save_images


def print_metrics(real_images, predict_images, model):
    print(type(model).__name__)
    print("PSNR:", mean_psnr(predict_images, real_images))
    print("SSIM:", mean_ssim(predict_images, real_images))
    print("FID:", get_fid(predict_images, real_images))


def REDNet_train(train_generator, val_generator, weights_path, lr=1e-4, epochs=20, finetuned=False, train=True):
    im_channels = 3
    im_size = 256
    model = REDNet(num_layers=7, channels=im_channels)
    loss = keras.losses.MeanSquaredError()

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss=loss)
    model.build((None, im_size, im_size, im_channels))
    model.summary()
    if finetuned:
        model.load_weights(get_latest_filename(weights_path))
        weights_path = weights_path + '-finetuned'

    if train:
        callbacks = get_default_callbacks(weights_path, train_generator, model)
        history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks)
        plot_graphs(history.history)

    model.load_weights(get_latest_filename(weights_path))
    return model


def MIRNET_train(train_generator, val_generator, weights_path, lr=1e-4, epochs=5, finetuned=False, train=True):
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model = mirnet_model(num_rrg=3, num_mrb=2, channels=32)
    model.compile(
        optimizer=optimizer, loss=charbonnier_loss, metrics=[peak_signal_noise_ratio]
    )
    model.summary()

    if finetuned:
        model.load_weights(get_latest_filename(weights_path))
        weights_path = weights_path + '-finetuned'

    if train:
        callbacks = get_default_callbacks(weights_path, train_generator, model)
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_peak_signal_noise_ratio",
                    factor=0.5,
                    patience=5,
                    verbose=1,
                    min_delta=1e-7,
                    mode="max",
                ), *callbacks
            ],
        )
        plot_graphs(history.history)

    model.load_weights(get_latest_filename(weights_path))
    return model


def train():
    im_size = 256
    batch_size = 1
    train_dirs = ['../datasets/comporation_datasets/imagenet/train',
                  '../datasets/comporation_datasets/allweather/train_test/train',
                  '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/train']
    test_dirs = ['../datasets/comporation_datasets/imagenet/test',
                 '../datasets/comporation_datasets/allweather/train_test/test',
                 '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/test']

    model_name = "rednet"
    dataset_inn = 1
    # model_name = "mirnet"
    data_dir = os.path.join(train_dirs[dataset_inn], 'fine')
    target_dir = os.path.join(train_dirs[dataset_inn], 'noisy')
    train_generator, val_generator = create_image_to_image_generator([data_dir, target_dir],
                                                                     batch_size=batch_size, seed=12345,
                                                                     im_size=im_size, ninty_rotate=False,
                                                                     vertical_flip=False, color_mode='rgb')

    dataset_inn = 1
    data_dir = os.path.join(test_dirs[dataset_inn], 'fine')
    target_dir = os.path.join(test_dirs[dataset_inn], 'noisy')
    test_generator, _ = create_image_to_image_generator([data_dir, target_dir],
                                                        batch_size=batch_size, seed=12345,
                                                        im_size=im_size, ninty_rotate=False,
                                                        vertical_flip=False, color_mode='rgb', validation_split=0)

    dataset_name = data_dir.split('/')[3]
    model = REDNet_train(train_generator, val_generator, f"../models/compare/{model_name}-{dataset_name}")
    # model = MIRNET_train(train_generator, val_generator, f"../models/compare/{model_name}-{dataset_name}", epochs=5)

    images, real = get_gen_images(test_generator, 500)
    test = model.predict(images)
    unionTestImages(images, test, path=f"../results/compare/{model_name}-{dataset_name}_union")
    save_images(np.concatenate([images, real, test], axis=2), path=f"../results/compare/{model_name}-{dataset_name}")

    print_metrics(real, test, model)


def fine_tune():
    im_size = 256
    batch_size = 1
    train_dirs = ['../datasets/comporation_datasets/imagenet/train',
                  '../datasets/comporation_datasets/allweather/train_test/train',
                  '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/train']
    test_dirs = ['../datasets/comporation_datasets/imagenet/test',
                 '../datasets/comporation_datasets/allweather/train_test/test',
                 '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/test']

    # model_name = "rednet"
    model_name = "mirnet"
    dataset_inn = 2
    data_dir = os.path.join(train_dirs[dataset_inn], 'fine')
    target_dir = os.path.join(train_dirs[dataset_inn], 'noisy')
    train_generator, val_generator = create_image_to_image_generator([data_dir, target_dir],
                                                                     batch_size=batch_size, seed=12345,
                                                                     im_size=im_size, ninty_rotate=False,
                                                                     vertical_flip=False, color_mode='rgb')

    data_dir = os.path.join(test_dirs[dataset_inn], 'fine')
    target_dir = os.path.join(test_dirs[dataset_inn], 'noisy')
    test_generator, _ = create_image_to_image_generator([data_dir, target_dir],
                                                        batch_size=batch_size, seed=12345,
                                                        im_size=im_size, ninty_rotate=False,
                                                        vertical_flip=False, color_mode='rgb', validation_split=0)

    dataset_name = data_dir.split('/')[3]
    # model = REDNet_train(train_generator, val_generator, f"../models/compare/{model_name}-imagenet", finetuned=True,
    #                      epochs=5, lr=1e-5)
    model = MIRNET_train(train_generator, val_generator, f"../models/compare/{model_name}-imagenet", finetuned=True,
                         epochs=1, lr=1e-5)

    images, real = get_gen_images(test_generator, 500)
    test = model.predict(images)
    unionTestImages(images, test, path=f"../results/compare/{model_name}-{dataset_name}-finetuned_union")
    save_images(np.concatenate([images, real, test], axis=2),
                path=f"../results/compare/{model_name}-{dataset_name}-finetuned")

    print_metrics(real, test, model)


def test():
    im_size = 256
    batch_size = 1
    train_dirs = ['../datasets/comporation_datasets/imagenet/train',
                  '../datasets/comporation_datasets/allweather/train_test/train',
                  '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/train']
    test_dirs = ['../datasets/comporation_datasets/imagenet/test',
                 '../datasets/comporation_datasets/allweather/train_test/test',
                 '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/test']

    # model_name = "rednet"
    model_name = "mirnet"
    for dataset_inn in range(0, 3):
        data_dir = os.path.join(train_dirs[dataset_inn], 'fine')
        train_name = data_dir.split('/')[3]
        # model = REDNet_train(None, None, f"../models/compare/{model_name}-{train_name}", train=False)
        model = MIRNET_train(None, None, f"../models/compare/{model_name}-{train_name}", train=False)

        for test_inn in range(0, 3):
            if test_inn == dataset_inn:
                continue
            data_dir = os.path.join(test_dirs[test_inn], 'fine')
            target_dir = os.path.join(test_dirs[test_inn], 'noisy')
            test_name = data_dir.split('/')[3]
            test_generator, _ = create_image_to_image_generator([data_dir, target_dir],
                                                                batch_size=batch_size, seed=12345,
                                                                im_size=im_size, ninty_rotate=False,
                                                                vertical_flip=False, color_mode='rgb',
                                                                validation_split=0)

            images, real = get_gen_images(test_generator, 500)
            test = model.predict(images)
            print(train_name, test_name)
            print_metrics(real, test, model)


if __name__ == "__main__":
    fine_tune()
