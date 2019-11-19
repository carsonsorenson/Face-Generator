import itertools
import cv2
import numpy as np
from load_attributes import load_attributes_wrapper

def normal_to_image(images):
    return [((image + 1.0) * 127.5).astype(np.uint8) for image in images]


def build_image_matrix(images, grid_size, output_file):
    imgs = normal_to_image(images)
    img_h, img_w, img_c = imgs[0].shape

    margin_x = 5
    margin_y = 5

    image_matrix = np.zeros((img_h * grid_size + margin_y * (grid_size - 1),
                             img_w * grid_size + margin_x * (grid_size - 1),
                             img_c), np.uint8)
    image_matrix.fill(255)

    positions = itertools.product(range(grid_size), range(grid_size))
    for (x_i, y_i), img in zip(positions, imgs):
        x = x_i * (img_w + margin_x)
        y = y_i * (img_h + margin_y)
        image_matrix[y:y+img_h, x:x+img_w, :] = img
    cv2.imwrite(output_file, image_matrix)


def load_real_images(data_dir, image_size, dataset_size, attributes=None):
    files = load_attributes_wrapper(data_dir, attributes)
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

