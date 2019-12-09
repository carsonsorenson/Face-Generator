import tensorflow as tf
import os
from save_results import normal_to_image, get_collage
from models import Models
from training import model_inputs, model_loss
import numpy as np


class Graph:
    def __init__(self, name, flags):
        self.graph = tf.Graph()
        self.flags = flags
        self.name = name
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            with self.sess.as_default():
                self.models = Models(self.flags.momentum, self.flags.init_weight_stddev, flags.epsilon, flags.image_size)
                self.input_real, self.input_z, self.lr = model_inputs(flags.image_size, flags.noise_size)
                self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, self.models)
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, os.path.join(flags.model_directory, self.name + '.ckpt'))

    def single_image(self, z):
        samples = self.generate(z)
        output, _ = self.discriminate(samples)
        best = np.where(output == max(output))[0][0]
        image = normal_to_image(samples[0][best])
        return image

    def collage(self, z):
        samples = self.generate(z)
        images = get_collage(samples, self.flags.grid_size)
        return images

    def generate(self, z):
        with self.graph.as_default():
            samples = self.sess.run(self.models.generator(self.input_z, False), feed_dict={self.input_z: z})
            return samples

    def discriminate(self, samples):
        with self.graph.as_default():
            output, _ = self.sess.run(self.models.discriminator(self.input_real, True), feed_dict={self.input_real: samples})
            return output
