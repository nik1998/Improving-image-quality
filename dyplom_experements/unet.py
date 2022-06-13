from utils.mykeras_utils import *
from keras import layers
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def get_unet_model(img_size, num_classes=1, channels=1):
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

    outputs = layers.Conv2D(num_classes, 3, activation="sigmoid", padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model


def create_image_to_image_generator(image_dirs: list,
                                    batch_size=8, im_size=128, color_mode='grayscale'):
    seed = random.randint(0, 2 ** 30)
    train_gens = []
    val_gens = []
    for i, path in enumerate(image_dirs):
        aug = AugmentationUtils() \
            .rescale() \
            .validation_split()
        train_generator, val_generator = aug.train_val_generator(path,
                                                                 target_size=(im_size, im_size),
                                                                 batch_size=batch_size,
                                                                 color_mode=color_mode,
                                                                 class_mode=None,
                                                                 seed=seed)
        train_gens.append(train_generator)
        val_gens.append(val_generator)

    train_generator = UnionGenerator(train_gens)
    val_generator = UnionGenerator(val_gens)
    return train_generator, val_generator


def train():
    keras.backend.clear_session()
    np.random.seed(12345)
    tf.random.set_seed(12345)
    # data_dir = '../datasets/not_sem/cats/real'
    # mask_dir = '../datasets/not_sem/cats/masks'
    data_dir = '../datasets/not_sem/OCT_dataset/real'
    mask_dir = '../datasets/not_sem/OCT_dataset/masks'
    img_size = (256, 256)
    num_classes = 1
    batch_size = 8

    # extend_func = lambda aug: aug.add_median_blur(p=0.5).add_gaussian_blur(p=0.5)

    train_generator, val_generator = create_image_to_image_generator([data_dir, mask_dir],
                                                                     batch_size=batch_size,
                                                                     im_size=img_size[0])
    check_not_interception(train_generator.generators[0].filenames, train_generator.generators[1].filenames)
    check_not_interception(train_generator.generators[1].filenames, train_generator.generators[0].filenames)
    check_not_interception(val_generator.generators[0].filenames, val_generator.generators[1].filenames)
    check_not_interception(val_generator.generators[1].filenames, val_generator.generators[0].filenames)
    test_generator("../results/test", train_generator, 50)
    model = get_unet_model(img_size, num_classes)
    # model.summary()
    # print(check_not_interception(train_generator.generators[0].filenames,train_generator.generators[1].filenames))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy",#dice_coef_loss,
                  metrics=['accuracy', f1_score])

    callbacks = get_default_callbacks("../models/unet_oct", val_generator, model, monitor_loss='val_f1_score',
                                      mode='max')

    epochs = 50
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks)
    plot_graphs(history.history)
    model.load_weights(get_latest_filename("../models/unet_oct/"))

    true_imgs, real_masks = get_gen_images(val_generator, 100)
    predicted_masks = model.predict(true_imgs)
    predicted_masks = simple_boundary(predicted_masks)
    print(f1_score(real_masks, predicted_masks))
    imgs = np.concatenate([true_imgs, real_masks, predicted_masks], axis=2)
    save_images(imgs, "../results/unet_oct/debug_imgs")


if __name__ == '__main__':
    train()
