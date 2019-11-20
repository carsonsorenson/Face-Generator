import tensorflow as tf
from load_images import load_real_images, load_fake_images, build_image_matrix
from flags import FLAGS
from models import Models
import random
import matplotlib.pyplot as plt
import os


def model_optimizers(d_loss, g_loss, lr, beta1):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [op for op in update_ops if op.name.startswith('generator') or op.name.startswith('discriminator')]

    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).minimize(g_loss, var_list=g_vars)
    return d_train_opt, g_train_opt


def model_loss(input_real, input_z, label_smoothing, models):
    # The loss of the generator
    g_model = models.generator(input_z, True)

    # The loss of discriminator when using real images
    d_model_real, d_logits_real = models.discriminator(input_real, reuse=False)

    # The loss of the discrimator when using fake images
    d_model_fake, d_logits_fake = models.discriminator(g_model, reuse=True)

    # calculate the loss for the real and fake images
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_real,
            labels=tf.ones_like(d_model_real) * label_smoothing
        )
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake,
            labels=tf.zeros_like(d_model_fake)
        )
    )
    # the total loss is the two added together
    d_loss = d_loss_real + d_loss_fake

    # calculate the loss for the generator
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake,
            labels=tf.ones_like(d_model_fake) * label_smoothing
        )
    )

    return d_loss, g_loss


# create tensors for model
def model_inputs(real_dim, z_dim):
    input_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    learning_rate = tf.placeholder(tf.float32, name='lr')
    return input_real, input_z, learning_rate


def save_plots(d_losses, g_losses, plot_directory):
    plt.plot(d_losses, label='Discriminator', alpha=0.6)
    plt.plot(g_losses, label='Generator', alpha=0.6)
    plt.title("Losses")
    plt.legend()
    plt.savefig(os.path.join(plot_directory, 'losses.png'))
    plt.show()
    plt.close()


def save_collage(sess, epoch, input_z, collage_directory, grid_size, models):
    test_z = load_fake_images(grid_size * grid_size, input_z.get_shape().as_list()[-1])
    samples = sess.run(models.generator(input_z, False), feed_dict={input_z: test_z})

    name = 'collage_' + str(epoch) + '.png'
    build_image_matrix(samples, grid_size, os.path.join(collage_directory, name))



def train():
    # load flags
    flags = FLAGS()

    # load our models
    models = Models(flags.momentum, flags.init_weight_stddev, flags.epsilon, flags.image_size)

    # get the tensorflow inputs
    input_real, input_z, learning_rate = model_inputs((flags.image_size, flags.image_size, 3), flags.noise_size)

    # get our loss variables
    d_loss, g_loss = model_loss(input_real, input_z, flags.label_smoothing, models)

    # get our optimizers
    d_opt, g_opt = model_optimizers(d_loss, g_loss, flags.learning_rate, flags.beta1)

    attributes = {'male': True, 'eyeglasses': True}
    images = load_real_images(flags.data_directory, flags.image_size, flags.dataset_size, attributes=attributes)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        d_losses = []
        g_losses = []
        iteration = 0
        for epoch in range(1, flags.epochs + 1):
            random.shuffle(images)
            for i in range(len(images) // flags.batch_size):
                iteration += 1
                batch_images = images[i * flags.batch_size:(i + 1) * flags.batch_size]
                batch_z = load_fake_images(flags.batch_size, flags.noise_size)

                _ = sess.run(
                    d_opt,
                    feed_dict={input_real: batch_images,input_z: batch_z, learning_rate: flags.learning_rate}
                )
                _ = sess.run(
                    g_opt,
                    feed_dict={input_real: batch_images, input_z: batch_z, learning_rate: flags.learning_rate}
                )
                d_losses.append(d_loss.eval({input_z: batch_z, input_real: batch_images}))
                g_losses.append(g_loss.eval({input_z: batch_z}))

                remaining_files = len(images) - ((i + 1) * flags.batch_size)
                print("\rEpoch: " + str(epoch) +
                      ", iteration: " + str(iteration) +
                      ", d_loss: " + str(round(d_losses[-1], 3)) +
                      ", g_loss: " + str(round(g_losses[-1], 3)) +
                      ", percent: " + str(round((len(images) - remaining_files) / len(images) * 100, 2)) +
                      ", remaining files in batch: " + str(remaining_files)
                      , sep=' ', end=' ', flush=True)
            save_plots(d_losses, g_losses, flags.plot_directory)
            save_collage(sess, epoch, input_z, flags.collage_directory, flags.grid_size, models)


if __name__ == '__main__':
    train()
