import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for m in range(Wk):
                    img_i = i + k - Hk // 2
                    img_j = j + m - Wk // 2

                    if 0 <= img_i < Hi and 0 <= img_j < Wi:
                        out[i, j] += image[img_i, img_j] * kernel[Hk - 1 - k, Wk - 1 - m]

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    Hb = Hk // 2
    Wb = Wk // 2
    conv_img = zero_pad(image, Hb, Wb)
    kernel = np.flip(kernel)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(conv_img[i:i + Hk, j:j + Wk] * kernel)

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    Hi, Wi = image.shape
    r_kernel = np.flip(kernel)
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # Обеспечиваем нечётные размеры ядра
    if Hk % 2 == 0:
        r_kernel = np.vstack((r_kernel, np.zeros((1, Wk))))
        Hk += 1
    if Wk % 2 == 0:
        r_kernel = np.hstack((r_kernel, np.zeros((Hk, 1))))
        Wk += 1

    conv_img = zero_pad(image, Hk // 2, Wk // 2)
    padded_strided = np.lib.stride_tricks.as_strided(conv_img, (Hi, Wi, Hk, Wk), strides=conv_img.strides * 2)
    out = np.sum(padded_strided * r_kernel, axis=(2, 3))
    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_faster(f, np.flip(g))

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_faster(f, np.flip(g) - g.mean())

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    Hf, Wf = f.shape
    Hg, Wg = g.shape

    # Нормализация шаблона (g)
    g_mean = np.mean(g)
    g_std = np.std(g)
    g_normalized = (g - g_mean) / g_std

    # Добавление обводки
    padded_f = zero_pad(f, Hg // 2, Wg // 2)

    # Инициализация результата
    out = np.zeros((Hf, Wf))

    # Векторизованное вычисление
    for i in range(Hf):
        for j in range(Wf):
            # Вырезка фрагмента изображения
            f_slice = padded_f[i:i + Hg, j:j + Wg]

            # Нормализация фрагмента (f_slice)
            f_slice_mean = np.mean(f_slice)
            f_slice_std = np.std(f_slice)
            f_slice_normalized = (f_slice - f_slice_mean) / f_slice_std

            # Вычисление нормированной корреляции
            correlation = np.sum(f_slice_normalized * g_normalized)
            norm_coeff = np.std(f_slice) * np.std(g)  # Более простой расчет коэффициента нормализации
            if norm_coeff != 0:
                out[i, j] = correlation / norm_coeff
            else:
                out[i, j] = 0  # Обработка случая, когда стандартное отклонение равно нулю

    return out
