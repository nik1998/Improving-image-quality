import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def quadro(t, coeffs):
    return coeffs[0] + coeffs[1] * t + coeffs[2] * t * t


def residuals(coeffs, y, t):
    return y - quadro(t, coeffs)


# http://scipy-lectures.org/intro/summary-exercises/auto_examples/plot_optimize_lidar_complex_data_fit.html#sphx-glr-download-intro-summary-exercises-auto-examples-plot-optimize-lidar-complex-data-fit-py
def test1():
    x = np.arange(0.0, 7.0, 0.1)
    y = 2 * x * x + 5 * x + 1
    plt.plot(x, y)
    plt.show()
    k0 = np.zeros(3, dtype=float)
    k, flag = leastsq(residuals, k0, args=(y, x))
    print(k)


def test2():
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = 3 * X * X + 2 * Y * Y
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False)
    plt.show()
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    k0 = np.zeros(6, dtype=float)
    k, flag = leastsq(oct, k0, args=(Z, X, Y))
    print(k)


# TODO modify equation
# https://dsp.stackexchange.com/questions/1714/best-way-of-segmenting-veins-in-leaves
def second_curve(x, y, c):
    return c[0] + c[1] * x + c[2] * y + c[3] * x * y + c[4] * x * x + c[5] * y * y


def oct(c, z, x, y):
    return z - second_curve(x, y, c)


def ridge_signal(k, th=0.01):
    if k[4] < 0 and k[5] < 0:
        if abs(k[4] + k[5]) > th:
            return 1.0
    return 0.0


def nonlinear_regression(img):
    h, w = img.shape
    x = np.arange(0, h)
    y = np.arange(0, w)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    Z = img.flatten()
    k0 = np.zeros(6, dtype=float)
    k, flag = leastsq(oct, k0, args=(Z, X, Y))
    return k


def ridge_detector(img, fsize=7):
    h, w = img.shape
    ans = np.zeros(img.shape, np.float)
    im = np.pad(img, ((0, 7), (0, 7)), 'edge')
    for i in range(h):
        for j in range(w):
            k = nonlinear_regression(im[i:i + 7, j:j + 7])
            ans[i, j] = ridge_signal(k)
    return ans


if __name__ == "__main__":
    test2()
