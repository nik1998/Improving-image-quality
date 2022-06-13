from utils.img_filters import *
import random


def apply_noise(im, stdNorm=True):
    image = np.reshape(im, im.shape[:-1])
    if stdNorm:
        image = std_norm_reverse(image)
    rnd = random.Random()
    j = rnd.randint(0, 3)
    if j == 0:
        image = noisy_with_defaults(image, "gauss")
    elif j == 1:
        image = noisy_with_defaults(image, "s&p")
    elif j == 2:
        image = noisy_with_defaults(image, "big_defect")
    elif j == 3:
        image = noisy_with_defaults(image, "light_side")
    image = np.clip(image, 0.0, 1.0)
    if stdNorm:
        image = std_norm_x(image)
    return np.expand_dims(image, axis=-1)


def noisy_with_defaults(image, noise_typ):
    rnd = random.Random()
    if noise_typ == "gauss":
        return gauss_noise(image, 0, 0.1)
    elif noise_typ == "s&p":
        return salt_paper(image, 0.5, 0.05)
    elif noise_typ == "light_side":
        return light_side(image, 1.5)
    elif noise_typ == "big_defect":
        count = rnd.randint(20, 100)
        return big_own_defect(image, count, hl=5, hr=15, wl=5, wr=15)


def gauss_noise(image, mean=0, var=0.1, p=1.0):
    rnd = random.Random()
    rnd = np.random.default_rng(rnd.randint(0, 2 ** 30))
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return image
    sigma = var ** 0.5
    gauss = rnd.normal(mean, sigma, image.shape)
    noisy = image + gauss
    return noisy.clip(0, 255)


def salt_paper(image, s_vs_p, amount, p=1.0):
    rnd = random.Random()
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return image
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [[rnd.randint(0, i - 1) for _ in range(int(num_salt))]
              for i in image.shape]
    image[tuple(coords)] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [[rnd.randint(0, i - 1) for _ in range(int(num_pepper))]
              for i in image.shape]
    image[tuple(coords)] = 0
    return image


def light_side(image, coeff, exponential=False, dist=10, p=1.0):
    rnd = random.Random()
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return image
    h, w, *_ = image.shape
    grim = np.mgrid[0:h, 0:w].astype(np.float)[0]
    if len(image.shape) == 3:
        grim = np.expand_dims(grim, axis=-1)
    if exponential:
        grim = np.exp(grim) * coeff
        grim = grim / np.max(grim)
    else:
        g2 = np.square(grim + dist)
        grim = coeff * h / g2

    j = rnd.randint(0, 4)
    if j == 1:
        grim = np.flipud(grim)
    elif j == 2:
        grim = np.transpose(grim, axes=(1, 0, 2))
    elif j == 3:
        grim = np.fliplr(np.transpose(grim, axes=(1, 0, 2)))
    im = image + grim
    return np.clip(im, 0.0, 1.0)


def big_own_defect(image, count, hl=5, hr=15, wl=5, wr=15, p=1.0):
    rnd = random.Random()
    rnd2 = np.random.default_rng(rnd.randint(0, 2 ** 30))
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return image
    mean = np.mean(image)
    for i in range(count):
        h = rnd.randint(hl, hr)
        w = rnd.randint(wl, wr)
        scalex = (w - 1) / 6  # standard deviation and tree sigma rule
        scaley = (h - 1) / 6
        m = np.zeros((h, w), dtype=np.float)

        # real a and b: a = ax* scale + loc, b = bx* scale + loc (ax,bx - sent parameters)
        # X = truncnorm(a=0, b=(w - 1), scale=scalex).rvs(size=3 * h * w)
        # X = X.round().astype(int)
        # Y = truncnorm(a=0, b=(h - 1), scale=scaley).rvs(size=3 * h * w)
        # Y = Y.round().astype(int)
        X = rnd2.normal(scale=scalex, loc=(w - 1) / 2, size=(3 * h * w))
        X = np.clip(X, 0.0, w - 1)
        X = X.round().astype(int)
        Y = rnd2.normal(scale=scaley, loc=(h - 1) / 2, size=(3 * h * w))
        Y = np.clip(Y, 0.0, h - 1)
        Y = Y.round().astype(int)
        for j in range(3 * h * w):
            m[Y[j], X[j]] += 1
        m = m / np.max(m) * mean

        ii = rnd.randint(0, image.shape[0] - h - 1)
        jj = rnd.randint(0, image.shape[1] - w - 1)
        image[ii:ii + h, jj: jj + w] += np.expand_dims(m, axis=-1)
    return np.clip(image, 0.0, 1.0)


def expansion_algorithm(image, count, sizel=10, sizer=50, gauss=True, p=1.0):
    rnd = random.Random()
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return image
    h, w, *_ = image.shape
    z = np.zeros(image.shape, dtype=np.float64)
    for _ in range(count):
        i = rnd.randint(0, h - 1)
        j = rnd.randint(0, w - 1)
        size = rnd.randint(sizel, sizer)
        pos = [(i, j)]
        cur = 1
        while cur < size:
            inn = rnd.randint(0, len(pos) - 1)
            ii, jj = pos[inn]
            if ii + 1 < h:
                pos.append((ii + 1, jj))
            if jj - 1 >= 0:
                pos.append((ii, jj - 1))
            if ii - 1 >= 0:
                pos.append((ii - 1, jj))
            if jj + 1 < w:
                pos.append((ii, jj + 1))
            z[ii, jj] += 1
            cur += 1
    z = z / np.max(z)
    if gauss:
        cv2.GaussianBlur(z, (0, 0), 1.0, z)
    else:
        z = unsharp_masking(z)
    image = image + z
    return np.clip(image, 0.0, 1.0)


def big_light_hole(img, count=3, hl=10, hr=30, wl=10, wr=30, p=1.0):
    rnd = random.Random()
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return img
    image = img.copy()
    mean = np.mean(image)
    delta = 0.1
    m = np.zeros(image.shape, dtype=np.float)
    for _ in range(count):
        h = rnd.randint(hl, hr)
        w = rnd.randint(wl, wr)
        h += (h + 1) % 2
        w += (w + 1) % 2

        ch = h // 2
        cw = w // 2
        ii = rnd.randint(h // 2, image.shape[0] - h // 2 - 1)
        jj = rnd.randint(w // 2, image.shape[1] - w // 2 - 1)

        for i in range(ii - h // 2, ii + h // 2 + 1):
            for j in range(jj - w // 2, jj + w // 2 + 1):
                d = (ii - i) ** 2 / ch / ch + (jj - j) ** 2 / cw / cw
                if d <= 1:
                    m[i, j] = mean + (rnd.randint(0, 1) % 2) * delta
    cv2.GaussianBlur(m, (0, 0), 2, m)
    image += 0.3 * m
    return np.clip(image, 0.0, 1.0)


def median_blur(image, k=5, p=1.0):
    rnd = random.Random()
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return image
    img = np.ascontiguousarray(image, dtype=np.float32)
    cv2.medianBlur(img, k, img)
    return img


def gaussian_blur(image, sigma=1.0, p=1.0):
    rnd = random.Random()
    pr = rnd.uniform(0.0, 1.0)
    if pr > p:
        return image
    cv2.GaussianBlur(image, (0, 0), sigma, image)
    return image
