import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_images_and_masks(image_dir, mask_dir):
    """Загружает изображения и соответствующие маски из указанных директорий"""
    images = []
    masks = []
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        images.append(image)
        masks.append(mask)

    return images, masks


def extract_tail(image, mask):
    """Извлекает область хвоста из изображения на основе маски"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    tail = image[y:y + h, x:x + w]
    tail_mask = mask[y:y + h, x:x + w]

    return tail, tail_mask

def normalize_tail(tail, tail_mask, target_size=(128, 128)):
    """Нормализует извлеченный хвост."""
    contours, _ = cv2.findContours(tail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), (MA, ma), angle = cv2.fitEllipse(largest_contour)

    rows, cols, _ = tail.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_tail = cv2.warpAffine(tail, M, (cols, rows))
    rotated_mask = cv2.warpAffine(tail_mask, M, (cols, rows))

    contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_tail = rotated_tail[y:y + h, x:x + w]
    normalized_tail = cv2.resize(cropped_tail, target_size)

    return normalized_tail


def calculate_iou(mask1, mask2):
    """Вычисляет метрику IoU между двумя масками"""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def plot_results(image, mask, tail, tail_mask, normalized_tail, gt_mask, iou_score, index):
    """Визуализирует исходное изображение, маску, извлеченный хвост, маску хвоста,
    нормализованный хвост и истинную маску с указанием IoU"""
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Исходное изображение')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Исходная маска')
    axes[0, 1].axis('off')

    if tail is not None and tail_mask is not None:
        axes[0, 2].imshow(cv2.cvtColor(tail, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('Вырезанный хвост')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(tail_mask, cmap='gray')
        axes[1, 0].set_title('Маска вырезанного хвоста')
        axes[1, 0].axis('off')
    else:
        axes[0, 2].set_title('Вырезанный хвост (None)')
        axes[0, 2].axis('off')

        axes[1, 0].set_title('Маска вырезанного хвоста (None)')
        axes[1, 0].axis('off')

    axes[1, 1].imshow(normalized_tail)
    axes[1, 1].set_title('Нормализованный хвост')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(gt_mask, cmap='gray')
    axes[1, 2].set_title(f'Истинная маска\nIoU: {iou_score:.2f}')
    axes[1, 2].axis('off')

    plt.suptitle(f'Изображение {index}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
