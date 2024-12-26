import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


def preprocess_image(image):
    # Переводим в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Применяем гистограммную эквализацию для улучшения контраста
    gray_image = cv2.equalizeHist(gray_image)

    # Уменьшаем шум с помощью гауссова размытия
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    return blurred_image


def compute_hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """Вычисляет HOG признаки для изображения и улучшает контраст HOG изображения для лучшей визуализации"""
    fd, hog_image = hog(
        image, orientations=orientations, pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, visualize=True
    )
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return fd, hog_image_rescaled


def cross_correlation(image, template, template_hog, template_shape):
    preprocessed_image = preprocess_image(image)

    result = cv2.matchTemplate(preprocessed_image, template, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    window = preprocessed_image[max_loc[1]:max_loc[1]+template_shape[0], max_loc[0]:max_loc[0]+template_shape[1]]
    if window.shape == template_shape:
        window_hog, _ = compute_hog(window)
        score = np.sum(window_hog - template_hog)
    else:
        score = -1

    return max_loc, max_val, score
