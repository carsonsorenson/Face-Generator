import os
import math


class FLAGS:
    def __init__(self, train=True):
        self.train = train
        self.epsilon = 0.00005
        self.init_weight_stddev = 0.02
        self.momentum = 0.9
        self.noise_size = 100
        self.lr = 0.0002
        self.batch_size = 128
        self.image_size = 64
        self.beta1 = 0.5
        self.dataset_size = math.inf
        self.epochs = 60
        self.project_directory = os.getcwd()
        self.model_directory = os.path.join(self.project_directory, 'models')
        self.grid_size = 8

        if self.train:
            self.data_directory = os.path.join(self.project_directory, 'data')
            self.output_image_directory = os.path.join(self.project_directory, 'samples')
            self.plot_directory = os.path.join(self.output_image_directory, 'plots')
            self.collage_directory = os.path.join(self.output_image_directory, 'collage')
            self.fixed_z_directory = os.path.join(self.output_image_directory, 'fixed')
            self.fixed_frequency = 100
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
