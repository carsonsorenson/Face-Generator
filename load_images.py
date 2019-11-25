import cv2
import numpy as np
from load_attributes import load_attributes_wrapper


def load_real_images(data_dir, image_size, dataset_size, attributes=None):
    files = load_attributes_wrapper(data_dir, attributes)
    print(len(files))
    images = []
    for i, file in enumerate(files):
        if i == dataset_size:
            break
        else:
            img = cv2.imread(file)
            cropped_img = img[55:55+128, 30:30+128]
            resized_im = cv2.resize(cropped_img, (image_size, image_size))
            normalized_img = (resized_im / 127.5) - 1.0
            images.append(normalized_img)
    return images


def load_fake_images(num_images, z_dim):
    return np.random.uniform(-1, 1, size=[num_images, z_dim])

