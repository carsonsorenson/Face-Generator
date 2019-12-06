import tensorflow as tf
from load_images import load_fake_images
from save_results import normal_to_image
from models import Models
from training import model_inputs, model_loss
from flags import FLAGS
import os
import cv2
import matplotlib.pyplot as plt

flags = FLAGS(visualize_progress=False)


class CreateGraph:
    def __init__(self, name):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            with self.sess.as_default():
                self.models = Models(flags.momentum, flags.init_weight_stddev, flags.epsilon, flags.image_size)
                self.input_real, self.input_z, self.lr = model_inputs(flags.image_size, flags.noise_size)
                self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, self.models)
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, os.path.join(flags.model_directory, name + '.ckpt'))

    def run(self, z):
        with self.graph.as_default():
            samples = self.sess.run(self.models.generator(self.input_z, False), feed_dict={self.input_z: z})
            img = normal_to_image(samples)[0]
            img = cv2.resize(img, (256, 256))
            return img


def generate():
    options = [
        'all',
        'male',
        'female',
        'black_hair_male',
        'blonde_hair_female',
        'blonde_hair_male_smiling',
        'black_hair_female_glasses'
    ]
    input_string = ['(' + str(i) + '): ' + s.replace('_', ' ').title() for i, s in enumerate(options)]
    input_string.append('(Q): Quit')
    models = {key: CreateGraph(key) for key in options}

    while True:
        print("Select an image to generate")
        print('\n'.join(input_string))
        val = input('>> ')
        if val == 'q' or val == 'Q':
            break
        z = load_fake_images(1, flags.noise_size)
        try:
            model = options[int(val)]
            img = models[model].run(z)
            plt.axis('off')
            plt.grid(b=None)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show()
        except:
            print('invalid input, try again')


'''
def generate(model_name):
    flags = FLAGS(visualize_progress=False)
    models = Models(flags.momentum, flags.init_weight_stddev, flags.epsilon, flags.image_size)
    input_real, input_z, lr = model_inputs(flags.image_size, flags.noise_size)
    d_loss, g_loss = model_loss(input_real, input_z, models)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        model_path = os.path.join(flags.model_directory, model_name + '.ckpt')
        saver.restore(sess, model_path)
        while True:
            z = load_fake_images(1000, flags.noise_size)
            
            samples = sess.run(models.generator(input_z, False), feed_dict={input_z: z})
            output, logits = sess.run(models.discriminator(input_real, True), feed_dict={input_real: samples})

            imgs = normal_to_image(samples)
            best = np.where(output == max(output))[0][0]
            print(max(output), min(output))
            worst = np.where(output == min(output))[0][0]

            cv2.imshow('best', cv2.resize(imgs[best], (256, 256)))
            cv2.imshow('worst', cv2.resize(imgs[worst], (256, 256)))


            #img = cv2.resize(img, (256, 256))
            cv2.waitKey()
'''
