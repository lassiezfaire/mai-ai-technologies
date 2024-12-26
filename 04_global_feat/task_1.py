import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

def ion2e(s):
    """Преобразует расширение файла с .jpg на .png"""
    if s[-3:] == 'jpg':
        return s[:-3] + 'png'

def load_and_resize_images_and_masks(images_dir, masks_dir, target_size=(256, 256)):
    """Загружает и изменяет размер изображений и масок из указанных директорий"""
    images = {}
    masks = {}
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg"):
            new_filename = ion2e(filename)
            img_path = os.path.join(images_dir, filename)
            mask_path = os.path.join(masks_dir, new_filename)
            if os.path.exists(mask_path):
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

                images[filename] = img_resized
                masks[filename] = mask_resized
    return images, masks

def process_image(image, mask, target_size=(256, 256)):
    """Применяет маску к изображению, находит контуры, выравнивает изображение и
    возвращает обработанное изображение и маску"""
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intc(box)

    width, height = target_size
    src_pts = np.array(box, dtype="float32")
    dst_pts = np.array([[0, height], [0, 0], [width, 0], [width, height]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    aligned_tail = cv2.warpPerspective(masked_image, M, target_size)

    predicted_mask = cv2.warpPerspective(mask, M, target_size)
    return aligned_tail, predicted_mask

def calculate_iou(predicted_mask, ground_truth_mask):
    """Вычисляет метрику Intersection over Union (IoU) между двумя масками"""
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    return intersection / union if union != 0 else 0

def process_and_evaluate(images, masks, target_size=(256, 256)):
    """Обрабатывает изображения и маски, вычисляет IoU для каждого изображения и возвращает словарь с IoU значениями"""
    iou_scores = {}
    for filename, image in images.items():
        mask = masks.get(filename)
        if mask is None:
            continue

        processed_image, predicted_mask = process_image(image, mask, target_size)
        if processed_image is None or predicted_mask is None:
            iou_scores[filename] = None
            continue

        predicted_mask_binary = predicted_mask > 0
        ground_truth_binary = mask > 0

        iou_scores[filename] = calculate_iou(predicted_mask_binary, ground_truth_binary)

    return iou_scores


def display_images(images, processed_images, predicted_masks, num_images=5):
    """Отображает оригинальные изображения, обработанные изображения и предсказанные маски
    для заданного количества изображений"""
    filenames = list(images.keys())
    num_images = min(num_images, len(filenames))

    for i in range(num_images):
        filename = filenames[i]
        original = images[filename]
        processed = processed_images.get(filename)
        predicted_mask = predicted_masks.get(filename)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Оригинальное изображение")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        if processed is not None:
            plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            plt.title("Нормализованный хвост")

        plt.subplot(1, 3, 3)
        if predicted_mask is not None:
            plt.imshow(predicted_mask, cmap="gray")
            plt.title("Нормализованная маска")

        plt.show()


def process_and_save_with_display(images, masks, target_size=(256, 256)):
    """Обрабатывает изображения и маски, вычисляет IoU, сохраняет обработанные изображения и маски,
    а также возвращает словари с IoU значениями, обработанными изображениями и масками"""
    iou_scores = {}
    processed_images = {}
    predicted_masks = {}

    for filename, image in images.items():
        mask = masks.get(filename)
        if mask is None:
            continue

        processed_image, predicted_mask = process_image(image, mask, target_size)
        if processed_image is None or predicted_mask is None:
            iou_scores[filename] = None
            continue

        predicted_mask_binary = predicted_mask > 0
        ground_truth_binary = mask > 0
        iou_scores[filename] = calculate_iou(predicted_mask_binary, ground_truth_binary)

        processed_images[filename] = processed_image
        predicted_masks[filename] = predicted_mask

    return iou_scores, processed_images, predicted_masks
