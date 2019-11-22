import os
import math

class FLAGS:
    def __init__(self):
        self.project_directory = os.getcwd()
        self.data_directory = os.path.join(self.project_directory, 'data')
        self.model_directory = os.path.join(self.project_directory, 'models')
        self.output_image_directory = os.path.join(self.project_directory, 'samples')
        self.plot_directory = os.path.join(self.output_image_directory, 'plots')
        self.collage_directory = os.path.join(self.output_image_directory, 'collage')
        self.fixed_z_directory = os.path.join(self.output_image_directory, 'fixed')

        self.epsilon = 0.00005
        self.init_weight_stddev = 0.02
        self.momentum = 0.9
        self.noise_size = 100
        self.lr_generator = 0.0002
        self.lr_discriminator = 0.00005
        self.batch_size = 64
        self.epochs = 2000
        self.image_size = 64
        self.beta1 = 0.5
        self.label_smoothing = 0.9
        self.dataset_size = math.inf
        self.gird_size = 5
        self.fixed_frequency = 50
        self.fixed_amount = 10

        self.create_directories()

    def create_directories(self):
        if not os.path.isdir(self.model_directory):
            os.mkdir(self.model_directory)
        if not os.path.isdir(self.output_image_directory):
            os.mkdir(self.output_image_directory)
        if not os.path.isdir(self.plot_directory):
            os.mkdir(self.plot_directory)
        if not os.path.isdir(self.collage_directory):
            os.mkdir(self.collage_directory)
        if not os.path.isdir(self.fixed_z_directory):
            os.mkdir(self.fixed_z_directory)
