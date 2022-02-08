from utils.conv_mod import *
from timeit import timeit

if __name__ == "__main__":
    style = np.random.normal(0.0, 1.0, size=[10, 32]).astype('float32')
    out = np.random.normal(0.0, 1.0, size=[10, 64, 64, 32]).astype('float32')
    conv = Conv2DMod(filters=768, kernel_size=3, padding='same', kernel_initializer='he_uniform')
    x = conv([out, style])
    x2 = conv.call_for_cpu([out, style])
    x = x.numpy()
    x2 = x2.numpy()
    print(np.sum(np.abs(x - x2)))

    elapsed_time = timeit("conv([out, style])", number=1000, globals=globals()) / 1000
