from neural_networks.adain import NeuralStyleTransfer
from utils.mykeras_utils import *
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


def train():
    keras.backend.clear_session()
    data_dir = '../datasets/unet/small_scale/all_real_images'
    mask_dir = '../datasets/unet/small_scale/all_mask_images'
    img_size = (256, 256)
    num_classes = 1
    batch_size = 8

    style_model = NeuralStyleTransfer((*img_size, 3), alpha=0.01)
    style_model.load_weights(get_latest_filename("../models/adain/"))

    style_gen = AugmentationUtils().horizontal_flip().vertical_flip().rescale().create_generator("../datasets/style",
                                                                                                 batch_size=batch_size,
                                                                                                 target_size=img_size,
                                                                                                 color_mode='rgb')
    extend_func = lambda aug: aug.add_median_blur(p=0.5).add_gaussian_blur(p=0.5)
    # .add_style_network(style_model, style_images)
    train_generator, val_generator = create_image_to_image_generator([data_dir, mask_dir],
                                                                     aug_extension=[extend_func],
                                                                     batch_size=batch_size,
                                                                     im_size=img_size[0])

    # add adain stylization
    train_generator = train_generator.style_augment(style_model, style_gen)

    test_generator("../results/test", train_generator, 500)
    model = get_unet_model(img_size, num_classes)
    # model.summary()
    # print(check_not_interception(train_generator.generators[0].filenames,train_generator.generators[1].filenames))

    model.compile(optimizer="rmsprop", loss=dice_coef_loss,
                  metrics=['accuracy', f1_score, precision_score, recall_score])

    callbacks = get_default_callbacks("../models/unet", val_generator, model, monitor_loss='val_f1_score', mode='max')

    epochs = 30
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=callbacks)
    plot_graphs(history.history)
    model.load_weights(get_latest_filename("../models/unet/"))

    true_imgs, real_masks = get_gen_images(val_generator, 100)
    predicted_masks = model.predict(true_imgs)
    predicted_masks = simple_boundary(predicted_masks)
    print(f1_score(real_masks, predicted_masks))
    imgs = np.concatenate([true_imgs, real_masks, predicted_masks], axis=2)
    save_images(imgs, "../results/unet/debug_imgs")


if __name__ == '__main__':
    train()
