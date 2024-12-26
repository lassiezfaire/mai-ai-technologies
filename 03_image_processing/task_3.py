import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.patches as patches
import os

def load_image(image_path):
    """Загружает изображение и преобразует его в массив NumPy."""
    image = Image.open(image_path)
    return np.array(image)

def extract_cells(image_np, cell_size, cell_count):
    """Извлекает ячейки из изображения и визуализирует их на исходном изображении."""
    img_height, img_width, _ = image_np.shape
    cells = []
    extracted_cells = 0
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    for y in range(0, img_height, cell_size):
        for x in range(0, img_width, cell_size):
            if extracted_cells >= cell_count:
                break
            cell = image_np[y:y + cell_size, x:x + cell_size]
            if cell.shape[:2] == (cell_size, cell_size) and cell.mean() < 240:  # порог для фильтрации пустых участков
                cells.append(cell)
                extracted_cells += 1
                rect = patches.Rectangle((x, y), cell_size, cell_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

    plt.title("Исходное изображение с выделенными ячейками")
    plt.axis('off')
    plt.draw()  # Отрисовка фигуры
    fig.canvas.draw()
    image_with_cells = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image_with_cells = image_with_cells.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return image_with_cells, cells

def prepare_cells_for_mosaic(cells):
    """Подготавливает ячейки для создания мозаики."""
    if len(cells) > 16:
        return cells[:16]
    elif len(cells) < 16:
        return cells * (16 // len(cells)) + cells[:16 % len(cells)]
    return cells

def create_mosaic(cells, cell_size, padding=10):
    """Создает мозаику из ячеек."""
    grid_size = 4
    mosaic_size = grid_size * cell_size + (grid_size - 1) * padding
    mosaic = np.ones((mosaic_size, mosaic_size, 3), dtype=np.uint8) * 255  # фон белого цвета

    for idx, cell in enumerate(cells):
        i = idx // grid_size
        j = idx % grid_size
        y_start = i * (cell_size + padding)
        x_start = j * (cell_size + padding)
        mosaic[y_start:y_start + cell_size, x_start:x_start + cell_size] = cell

    return mosaic

def process_image(image_path, cell_size, cell_count, padding=10):
    """Обрабатывает одно изображение и возвращает изображение с выделенными ячейками и мозаику."""
    image_np = load_image(image_path)
    image_with_cells, cells = extract_cells(image_np, cell_size, cell_count)
    cells = prepare_cells_for_mosaic(cells)
    mosaic = create_mosaic(cells, cell_size, padding)
    return image_with_cells, mosaic

def display_all_images(folder_path, cell_size, cell_count, padding=10):
    """Обрабатывает все изображения в папке и отображает их на одном экране."""
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(image_files)

    plt.figure(figsize=(8, num_images * 2))

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image_with_cells, mosaic = process_image(image_path, cell_size, cell_count, padding)

        plt.subplot(num_images, 2, 2 * idx + 1)
        plt.imshow(image_with_cells)
        plt.title(f"Ячейки на изображении {os.path.splitext(image_file)[0]}")
        plt.axis('off')

        plt.subplot(num_images, 2, 2 * idx + 2)
        plt.imshow(mosaic)
        plt.title(f"Мозаика {os.path.splitext(image_file)[0]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
