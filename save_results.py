import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def normal_to_image(images):
    return [((image + 1.0) * 127.5).astype(np.uint8) for image in images]


def save_collage(images, grid_size, output_file):
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


def save_fixed_images(images, model_name, iteration, directory):
    images = normal_to_image(images)
    for index, image in enumerate(images):
        image = cv2.resize(image, (256, 256))
        filename = 'fixed_' + model_name + '_' + iteration + '_' + str(index) + '.png'
        path = os.path.join(directory, filename)
        cv2.imwrite(path, image)


def save_plots(d_losses, g_losses, plot_directory, model_name):
    plt.plot(d_losses, label='Discriminator', alpha=0.6)
    plt.plot(g_losses, label='Generator', alpha=0.6)
    plt.title("Losses")
    plt.legend()
    name = 'losses_' + model_name + '.png'
    plt.savefig(os.path.join(plot_directory, name))
    plt.show()
    plt.close()
