import tensorflow as tf
from keras import backend as K
from tensorflow.keras import layers

from utils.mykeras_utils import *

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_unet_model(img_size, num_classes, channels=1):
    inputs = keras.Input(shape=img_size + (channels,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


if __name__ == '__main__':
    keras.backend.clear_session()
    data_dir = "../datasets/unet/images"
    mask_dir = "../datasets/unet/mask"
    img_size = (128, 128)
    num_classes = 1
    batch_size = 16
    seed = random.randint(0, 2 ** 30)

    aug = AugmentationUtils() \
        .rescale() \
        .add_median_blur() \
        .add_gaussian_blur() \
        .validation_split()
    img_generator = aug.create_generator(data_dir, seed=seed, subset='training')
    img_generator2 = aug.create_generator(data_dir, seed=seed, subset='validation')

    # print(check_not_interception(img_generator.filenames, img_generator2.filenames))

    aug = AugmentationUtils() \
        .rescale() \
        .validation_split()

    mask_generator = aug.create_generator(mask_dir, seed=seed, subset='training')
    mask_generator2 = aug.create_generator(mask_dir, seed=seed, subset='validation')

    train_generator = UnionGenerator([img_generator, mask_generator], batch_size).reflect_rotate()
    val_generator = UnionGenerator([img_generator2, mask_generator2], batch_size).reflect_rotate()

    test_generator("../results/test", train_generator)
    # test_generator("../results/test/val", val_generator)

    # Build model
    model = get_unet_model(img_size, num_classes)
    # model.summary()

    model.compile(optimizer="rmsprop", loss="binary_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint("../models/unet/best_model.h5", save_best_only=True)
    ]

    epochs = 30
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks)
    plot_graphs(history.history)
    model.load_weights("../models/unet/best_model.h5")

    true_imgs = []
    real_masks = []
    batch_index = 0
    total = len(val_generator)

    for data in val_generator:
        true_imgs.extend(list(data[0]))
        real_masks.extend(list(data[1]))
        batch_index = batch_index + 1
        if batch_index > total:
            break
    real_masks = np.asarray(real_masks)
    predicted_masks = model.predict(real_masks)
    true_imgs = true_imgs[:len(predicted_masks)]
    # validation score
    # predicted_masks = np.reshape(predicted_masks, predicted_masks.shape[:-1])
    predicted_masks = simple_boundary(predicted_masks)
    imgs = []
    for im, im2, im3 in zip(true_imgs, real_masks, predicted_masks):
        img = np.hstack([im, im2, im3])
        imgs.append(img)
    imgs = np.asarray(imgs)
    save_images(imgs, "../results/unet/debug_imgs")
    true_imgs = np.asarray(true_imgs).flatten().astype(dtype=np.float32)
    print(f1_m(true_imgs, predicted_masks.flatten()))
    print(np.sum(np.abs(true_imgs - predicted_masks.flatten())) / true_imgs.shape[0])

    # compare
    # recursive_read_operate_save(input_dir, "../unet/result_images", check_model(model, simple_boundary))

    # real test
    # ans = process_real_frame(model, "../scan_images/", "../unet/scan_results", interpolate=True)
    # ans = simple_boundary(ans)
    # save_images(ans, "../unet/scan_results")
    # showImage(read_image(val_input_img_paths[0]))
    # showImage(predicted_masks[0])
