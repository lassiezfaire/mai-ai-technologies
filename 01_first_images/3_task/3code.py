import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_one_image(image: np.ndarray, title: str) -> None:
    """
    Отобразить изображение с помощью matplotlib.
    Вспомогательная функция

    :param image: изображение для отображения
    :param title: заголовок для изображения
    :return: None
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    plt.title(title)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.show()

def rotate(path_to_image, rotation_point: tuple, rotation_angle: float) -> np.ndarray:
    """
    Повернуть картинку по часовой стрелке на указанный угол без потери углов

    :param path_to_image: путь к картинке
    :param rotation_point: точка вращения картинки
    :param rotation_angle: угол вращения картинки
    :return: повёрнутая картинка
    """

    img = cv2.imread(cv2.samples.findFile(path_to_image))

    if img is None:
        sys.exit("Could not read the image.")


    height, width = img.shape[:2]

    corners = np.array(
        [
            [width - 1, height - 1],
            [0, 0],
            [width - 1, 0],
            [0, height - 1]
        ]
    )

    rotation_matrix = cv2.getRotationMatrix2D(rotation_point, rotation_angle, 1)

    rotation = np.hstack((corners, np.ones((corners.shape[0], 1))))

    rotated = rotation_matrix @ rotation.T

    rotation_matrix[:, 2] -= rotated.min(axis=1)

    new_size = np.int64(np.ceil(rotated.max(axis=1) - rotated.min(axis=1)))

    rotated_img = cv2.warpAffine(img, rotation_matrix, new_size)

    return rotated_img

path = r'images//lk.jpg'
result = rotate(path, rotation_point=(200, 200), rotation_angle=15)

plot_one_image(image=result, title="Rotated image")
