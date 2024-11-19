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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.show()

def find_road_number(path_to_image: str):
    """
    Поиск номера полосы, на которую нужно перестроиться

    :param path_to_image: путь к картинке с автомобилем
    :return: строка с результатом
    """

    img = cv2.imread(cv2.samples.findFile(path))

    if img is None:
        sys.exit("Could not read the image.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blue_lower = np.array([110, 50, 50], np.uint8)
    blue_upper = np.array([130, 255, 255], np.uint8)
    car_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    red_lower = np.array([0, 100, 100], np.uint8)
    red_upper = np.array([10, 255, 255], np.uint8)
    obstacle_mask = cv2.inRange(hsv, red_lower, red_upper)

    car_contour, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found_contours = np.zeros(img.shape, np.uint8)
    cv2.drawContours(found_contours, car_contour, -1, (0, 0, 255), 2)

    obstacle_contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(found_contours, obstacle_contours, -1, (255, 0, 0), 2)

    lanes_num = 0
    for obstacle in obstacle_contours:
        lanes_num += 1

    lanes_num += 1

    lane_width = img.shape[1] // lanes_num

    car_x = int(np.mean([c[0][0] for c in car_contour[0]]))
    car_lane = car_x // lane_width

    checked_lanes = []
    lane_obstacles = [False] * lanes_num
    for c in obstacle_contours:
        x, _, w, _ = cv2.boundingRect(c)
        obstacle_lane = x // lane_width
        if (obstacle_lane * lane_width) <= x <= ((obstacle_lane + 1) * lane_width) - w:
            lane_obstacles[obstacle_lane] = True

        for i in range(lanes_num):
            if not lane_obstacles[i]:
                if i != car_lane:
                    checked_lanes.append(i)

    if checked_lanes[-1] != car_lane:
        return f'Нужно перестроиться на полосу номер {checked_lanes[-1]}'
    else:
        return 'Перестраиваться не нужно'

path = r'images//image_00.jpg'
print(find_road_number(path))

path = r'images//image_01.jpg'
print(find_road_number(path))

path = r'images//image_02.jpg'
print(find_road_number(path))
