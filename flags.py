import os

class FLAGS:
    def __init__(self):
        self.epsilon = 0.00005
        self.init_weight_stddev = 0.02
        self.momentum = 0.9
        self.noise_size = 100
        self.learning_rate = 0.0002
        self.batch_size = 64
        self.epochs = 20
        self.image_size = 64
        self.beta1 = 0.5
        self.label_smoothing = 0.9
        self.dataset_size = 100000
        self.data_directory = './data'

        self.output_image_directory = './samples'

        self.plot_directory = os.path.join(self.output_image_directory, 'plots')

        self.collage_directory = os.path.join(self.output_image_directory, 'collage')
        self.grid_size = 5

        self.fixed_z_directory = os.path.join(self.output_image_directory, 'fixed')
        self.fixed_frequency = 50
        self.fixed_amount = 10

        if not os.path.isdir(self.data_directory):
            os.mkdir(self.data_directory)
            if not os.path.isdir(self.plot_directory):
                os.mkdir(self.plot_directory)
            if not os.path.isdir(self.collage_directory):
                os.mkdir(self.collage_directory)
            if not os.path.isdir(self.fixed_z_directory):
                os.mkdir(self.fixed_z_directory)