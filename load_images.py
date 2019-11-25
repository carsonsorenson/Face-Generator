import cv2
import numpy as np
from load_attributes import load_attributes_wrapper


def normal_to_image(images):
    return [((image + 1.0) * 127.5).astype(np.uint8) for image in images]


def build_image_matrix(images, grid_size, output_file):
    images = normal_to_image(images)
    image_height, image_width, image_channel = images[0].shape
    for img in images:
        assert img.shape == images[0].shape, "All images need to be the same dimension"

    # Put a little bit of whitespace around each image
    margin_x = 3
    margin_y = 3

    matrix_height = image_height * grid_size + margin_y * (grid_size - 1)
    matrix_width = image_width * grid_size + margin_x * (grid_size - 1)
    matrix = np.zeros((matrix_height, matrix_width, image_channel), np.uint8)
    matrix.fill(255)

    index = 0
    for i in range(grid_size):
        for j in range(grid_size):
            x = i * (image_width + margin_x)
            y = j * (image_height + margin_y)
            matrix[y:y+image_height, x:x+image_width, :] = images[index]
            index += 1
    cv2.imwrite(output_file, matrix)


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

