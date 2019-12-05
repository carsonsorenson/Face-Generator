import tensorflow as tf
from load_images import load_real_images, load_fake_images
from save_results import save_collage, save_fixed_images, save_plots
from models import Models
import random
import os

tf.logging.set_verbosity(tf.logging.ERROR)


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


def model_loss(input_real, input_z, models):
    # The loss of the generator
    g_model = models.generator(input_z, True)

    # The loss of discriminator when using real images
    d_model_real, d_logits_real = models.discriminator(input_real, reuse=False)

    # The loss of the discriminator when using fake images
    d_model_fake, d_logits_fake = models.discriminator(g_model, reuse=True)

    # calculate the loss for the real and fake images
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_real,
            labels=tf.ones_like(d_model_real) * random.uniform(0.8, 1.0)
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
            labels=tf.ones_like(d_model_fake)
        )
    )
    return d_loss, g_loss


def model_inputs(real_dim, z_dim):
    input_real = tf.placeholder(tf.float32, (None, real_dim, real_dim, 3), name='input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    lr = tf.placeholder(tf.float32, name='lr')
    return input_real, input_z, lr


def print_summary(epoch, iteration, d_losses, g_losses, percent, remaining_files):
    print("\rEpoch: " + str(epoch) +
          ", iteration: " + str(iteration) +
          ", d_loss: " + str(round(d_losses[-1], 3)) +
          ", g_loss: " + str(round(g_losses[-1], 3)) +
          ", epoch percent: " + str(round(percent, 2)) +
          ", files left in epoch: " + str(remaining_files)
          , sep=' ', end='', flush=True)


def train(flags, model_name, load=False, iteration=0, epoch=0, attributes=None):
    tf.reset_default_graph()

    # Load models, inputs, losses, and optimizers
    models = Models(flags.momentum, flags.init_weight_stddev, flags.epsilon, flags.image_size)
    input_real, input_z, lr = model_inputs(flags.image_size, flags.noise_size)
    d_loss, g_loss = model_loss(input_real, input_z, models)
    d_opt, g_opt = model_optimizers(d_loss, g_loss, flags.lr, flags.beta1)

    # load real and fake inputs into networks
    images = load_real_images(flags.data_directory, flags.image_size, flags.dataset_size, attributes)
    fixed_z = load_fake_images(flags.fixed_amount, flags.noise_size)

    num_batches = len(images) // flags.batch_size
    d_losses, g_losses = [], []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if load:
            model_path = os.path.join(flags.model_directory, model_name + '_' + str(epoch) + '.ckpt')
            saver.restore(sess, model_path)
        while epoch < flags.epochs:
            epoch += 1
            random.shuffle(images)
            for i in range(num_batches):
                iteration += 1
                batch_images = images[i * flags.batch_size:(i + 1) * flags.batch_size]
                batch_z = load_fake_images(flags.batch_size, flags.noise_size)

                d_feed_dict = {input_real: batch_images, input_z: batch_z, lr: flags.lr}
                g_feed_dict = {input_real: batch_images, input_z: batch_z, lr: flags.lr}

                _ = sess.run(d_opt, feed_dict=d_feed_dict)
                _ = sess.run(g_opt, feed_dict=g_feed_dict)

                d_losses.append(d_loss.eval({input_z: batch_z, input_real: batch_images}))
                g_losses.append(g_loss.eval({input_z: batch_z}))

                remaining_files = len(images) - ((i + 1) * flags.batch_size)
                percent = (len(images) - remaining_files) / len(images) * 100
                print_summary(epoch, iteration, d_losses, g_losses, percent, remaining_files)

                # Save the progress of our fixed noises to see how the model is updating
                if flags.visualize_progress and (iteration - 1) % flags.fixed_frequency == 0:
                    fixed_samples = sess.run(models.generator(input_z, False), feed_dict={input_z: fixed_z})
                    fixed_index = (iteration - 1) // flags.fixed_frequency
                    save_fixed_images(fixed_samples, model_name, fixed_index, flags.fixed_z_directory)

            if flags.visualize_progress:
                # save the loss plots
                save_plots(d_losses, g_losses, flags.plot_directory, model_name)

                # create images from generator to create a collage of images
                collage_size = flags.grid_size * flags.grid_size
                test_z = load_fake_images(collage_size, flags.noise_size)
                samples = sess.run(models.generator(input_z, False), feed_dict={input_z: test_z})
                name = 'collage_' + model_name + '_' + str(epoch).zfill(5) + '.png'
                save_collage(samples, flags.grid_size, os.path.join(flags.collage_directory, name))
            model_path = os.path.join(flags.model_directory, model_name + '_' + str(epoch) + '.ckpt')
            saver.save(sess, model_path)
