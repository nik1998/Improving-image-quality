from utils.mykeras_utils import AugmentationUtils
from utils.mylibrary import save_images, prepare_dataset

if __name__ == '__main__':
    im_size = 256
    aug = AugmentationUtils().rescale()
    generator = aug.create_generator("../datasets/comporation_datasets/imagenet/train/fine",
                                     target_size=(im_size, im_size),
                                     batch_size=1, shuffle=False,
                                     color_mode='rgb')

    #images = get_gen_images(generator)
    save_images(generator, "../datasets/comporation_datasets/imagenet/train/images", )
    # splitfolders.ratio('../datasets/comporation_datasets/allweather/fine',
    #                    output="../datasets/comporation_datasets/allweather/train_test", seed=1337, ratio=(.8, 0.0, 0.2))

    # prepare_dataset("../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train/fine",
    #                 '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/train/fine', 256, step=256,
    #                 drop=0,
    #                 determined=True, gray=False)
    # save_images.inn = 0
    # prepare_dataset("../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train/noisy",
    #                 '../datasets/comporation_datasets/SSID/SIDD_Medium_Srgb/train_test_split/train/noisy', 256, step=256,
    #                 drop=0,
    #                 determined=True, gray=False)
