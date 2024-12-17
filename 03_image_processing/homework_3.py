from glob import glob
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_mask(image, lower_bound, upper_bound):
    """Создаёт маску с помощью cv2"""
    return cv2.inRange(image, lower_bound, upper_bound)


def count_pixels(mask):
    """Считает площадь маски в пикселях"""
    return cv2.countNonZero(mask)


def classify_image(image_path, color_ranges):
    """Классифицирует изображение, используя заданные диапазоны цветов."""
    try:
        hsv_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)
        masks = {name: create_mask(hsv_image, lower, upper) for name, (lower, upper) in color_ranges.items()}
        counts = {name: count_pixels(mask) for name, mask in masks.items()}
        classification = max(counts, key=counts.get)
        return classification, hsv_image
    except cv2.error as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def display_images(image_dir, color_ranges):
    image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))  # предполагаем что картинки jpg

    num_images = len(image_paths)
    rows = math.ceil(num_images ** 0.5)
    cols = math.ceil(num_images / rows)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()

    for i, image_path in enumerate(image_paths):
        classification, image = classify_image(image_path, color_ranges)
        if image is not None:
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))  # преобразуем обратно в RGB для отображения
            axes[i].set_title(f"Class: {classification if classification else 'Error'}")
            axes[i].axis('off')

    # Удаляем лишние оси, если их количество больше, чем нужно
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


color_ranges = {
    "Forest": (np.array([35, 25, 25]), np.array([85, 255, 255])),
    "Desert": (np.array([15, 25, 25]), np.array([35, 255, 255])),
}

image_directory = "desert_forest"

display_images(image_directory, color_ranges)