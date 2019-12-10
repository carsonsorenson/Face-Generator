import cv2
import numpy as np
from load_attributes import load_attributes_wrapper


def load_image(image_path, image_size):
    img = cv2.imread(image_path)
    cropped_img = img[55:55 + 128, 30:30 + 128]
    resized_im = cv2.resize(cropped_img, (image_size, image_size))
    normalized_img = (resized_im / 127.5) - 1.0
    return normalized_img


def load_real_images(data_dir, image_size, dataset_size, attributes=None):
    files = load_attributes_wrapper(data_dir, attributes)
    print(len(files))
    images = []
    for i, file in enumerate(files):
        if i == dataset_size:
            break
        else:
            normalized_img = load_image(file, image_size)
            images.append(normalized_img)
    return images


def load_fake_images(num_images, z_dim):
    return np.random.normal(0, 1, size=[num_images, z_dim])

