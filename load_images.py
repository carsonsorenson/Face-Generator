import itertools
import cv2
import numpy as np
from load_attributes import load_attributes_wrapper


def build_image_matrix(images, grid_size, output_file):
    imgs = [cv2.imread(i) for i in images]
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


def load_real_images(data_dir, attributes=None):
    files = load_attributes_wrapper(data_dir, attributes)
    print(len(files))

    build_image_matrix(files[:64], 8, 'test.png')



def load_fake_images(num_images):
    pass

if __name__ == '__main__':
    load_real_images('./data/')


