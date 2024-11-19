import cv2
import sys

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

    ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.show()

def plot_maze_path(path_to_image: str):
    """
    Найти путь через лабиринт с помощью операций дилатации и эрозии

    :param path_to_image: путь к картинке с лабиринтом
    :return:
    """
    img = cv2.imread(cv2.samples.findFile(path_to_image))

    if img is None:
        sys.exit("Could not read the image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plot_one_image(image=img, title="OG maze image")

    ret, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV)
    # plot_one_image(image=binary_img, title="Binaried maze image")

    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solution = np.zeros(binary_img.shape, np.uint8)
    cv2.drawContours(solution, contours, 0, (255, 255, 255), 5)
    # plot_one_image(image=solution, title="Contours")


    kernel = np.ones((21, 21), np.uint8)
    dilated = cv2.dilate(solution, kernel, iterations=1)
    # plot_one_image(image=dilated, title="Applied dilation")

    erosion = cv2.erode(dilated, kernel, iterations=1)
    # plot_one_image(image=erosion, title="Applied erosion")

    diff = cv2.absdiff(erosion, dilated)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img[diff == 255] = (0, 255, 0)

    plot_one_image(image=img, title="Solved")

path = r'images//20 by 20 orthogonal maze.png'
plot_maze_path(path_to_image=path)

path = r'images//20 by 22 orthogonal maze.png'
plot_maze_path(path_to_image=path)

path = r'images//25 by 22 orthogonal maze.png'
plot_maze_path(path_to_image=path)

path = r'images//30 by 30 orthogonal maze.png'
plot_maze_path(path_to_image=path)

path = r'images//30 by 30 orthogonal maze.png'
plot_maze_path(path_to_image=path)
