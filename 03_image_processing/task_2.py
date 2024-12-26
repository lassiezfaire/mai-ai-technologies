import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_face_parts(img_path, face_cascade, eye_cascade, nose_cascade, mouth_cascade):
    """Ищет части лица и создаёт маску"""
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    mask = np.zeros_like(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        for cascade, label in [(eye_cascade, "eyes"), (nose_cascade, "nose"), (mouth_cascade, "mouth")]:
            parts = cascade.detectMultiScale(roi_gray)
            for (px, py, pw, ph) in parts:
                cv2.ellipse(mask, ((px + pw//2 + x), (py + ph//2 + y)), (pw//2, ph//2), 0, 0, 360, 255, -1)

    return img, mask

def apply_seamless_cloning(src_img_path, dst_img_path, face_cascade, eye_cascade, nose_cascade, mouth_cascade):
    """Применяет seamless cloning"""
    src_img, src_mask = detect_face_parts(src_img_path, face_cascade, eye_cascade, nose_cascade, mouth_cascade)
    dst_img, _ = detect_face_parts(dst_img_path, face_cascade, eye_cascade, nose_cascade, mouth_cascade)

    height, width = dst_img.shape[:2]
    src_resized = cv2.resize(src_img, (width, height))
    src_mask_resized = cv2.resize(src_mask, (width, height))

    center = (width // 2, height // 2)
    seamless_img = cv2.seamlessClone(src_resized, dst_img, src_mask_resized, center, cv2.NORMAL_CLONE)

    return src_img, dst_img, src_resized, seamless_img

def display_images(src_img, dst_img, src_resized, seamless_img):
    """Выводит изображения"""
    plt.figure(figsize=(8, 7))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB))
    plt.title('Source Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB))
    plt.title('Destination Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(src_resized, cv2.COLOR_BGR2RGB))
    plt.title('Resized Source Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(seamless_img, cv2.COLOR_BGR2RGB))
    plt.title('Seamless Cloned Image')
    plt.axis('off')

    plt.show()
