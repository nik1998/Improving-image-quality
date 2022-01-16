from stylegan import *
from utils.mykeras_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def brute_force_find_face(img_path, output, max_iter=500):
    im = read_image(img_path, gray=False)

    best_latent = noiseList(1)
    best_dist = 9e9

    model = StyleGAN(None, lr=0.0001, silent=False)
    model.load(20)

    n = nImage(1)

    for i in range(max_iter):
        print('Best ({}/{}) = {}'.format(i, max_iter, round(best_dist, 5)))
        if i < 10:
            latents = noiseList(1)
        else:
            latents = best_latent * 0.7 + noiseList(1) * 0.3
        image = model.GAN.GMA(latents + [n])[0]
        dist = ((image - im) ** 2).mean()
        if dist < best_dist:
            best_dist = dist
            best_latent = latents
            cv2.imwrite(output + 'img' + str(i) + '.jpg', image)
            np.save(output + 'best.npy', best_latent)


if __name__ == '__main__':
    brute_force_find_face("../datasets/test_sem_internet/test/img0.png", "../results/styleGAN/test/")
