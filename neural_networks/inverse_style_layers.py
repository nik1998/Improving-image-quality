from neural_networks.stylegan import *
from utils.mykeras_utils import *


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def brute_force_find_face(img_path, output, max_iter=1000):
    im = read_image(img_path, gray=False)

    best_latent = noiseList(1)
    best_dist = 9e9

    model = StyleGAN(None, lr=0.0001, silent=False)
    model.load(20)

    n = nImage(1)
    pr = None
    nl = None
    lr = 0.01
    for i in range(max_iter):
        print('Best ({}/{}) = {}'.format(i, max_iter, round(best_dist, 5)))
        if i < 50:
            latents = noiseList(1)
        else:
            if pr is not None:
                nl = pr
            else:
                nl = np.zeros_like(best_latent)
                i = random.randint(0, 511)
                nl[:, :, i] = (random.randint(0, 1) * 2 - 1) * lr #* (max_iter - i) / max_iter
            latents = list(np.asarray(best_latent) + np.asarray(nl))
        image = model.GAN.GMA(latents + [n])[0]
        dist = ((image.numpy() - im) ** 2).mean()
        if dist < best_dist and abs(dist - best_dist) > 1e-6:
            pr = nl
            best_dist = dist
            best_latent = latents
            np.save(output + 'best.npy', best_latent)
        else:
            pr = None
    image = model.GAN.GMA(best_latent + [n])[0]
    cv2.imwrite(output + 'img.jpg', image.numpy() * 255)


if __name__ == '__main__':
    brute_force_find_face("../datasets/test_sem_internet/test/img0.png", "../results/styleGAN/test/")
