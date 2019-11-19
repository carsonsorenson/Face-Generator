import tensorflow as tf

class Models:
    def __init__(self, momentum, init_weight_stddev, epsilon, image_size):
        self.momentum = momentum
        self.init_weight_stddev = init_weight_stddev
        self.epsilon = epsilon
        self.image_size = image_size

    def generator(self, z, training):
        with tf.variable_scope("generator", reuse=not training):
            s16 = self.image_size // 16
            fc = tf.layers.dense(z, self.image_size * 8 * s16 * s16)
            fc = tf.reshape(fc, (-1, s16, s16, self.image_size * 8))
            batch_fc = tf.layers.batch_normalization(inputs=fc,
                                                     training=training,
                                                     momentum=self.momentum,
                                                     epsilon=self.epsilon)
            fc_out = tf.nn.leaky_relu(batch_fc)

            # 4x4x512 -> 8x8x256
            trans_conv1 = tf.layers.conv2d_transpose(inputs=fc_out,
                                                     filters=self.image_size * 4,
                                                     kernel_size=(5, 5),
                                                     strides=(2, 2),
                                                     padding="same",
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         stddev=self.init_weight_stddev))
            batch_trans_conv1 = tf.layers.batch_normalization(inputs=trans_conv1,
                                                              momentum=self.momentum,
                                                              training=training,
                                                              epsilon=self.momentum)
            trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1)

            # 8x8x256 -> 16x16x128
            trans_conv2 = tf.layers.conv2d_transpose(inputs=trans_conv1_out,
                                                     filters=self.image_size * 2,
                                                     kernel_size=(5, 5),
                                                     strides=(2, 2),
                                                     padding="same",
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         stddev=self.init_weight_stddev))
            batch_trans_conv2 = tf.layers.batch_normalization(inputs=trans_conv2,
                                                              momentum=self.momentum,
                                                              training=training,
                                                              epsilon=self.momentum)
            trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2)

            # 16x16x128 -> 32x32x64
            trans_conv3 = tf.layers.conv2d_transpose(inputs=trans_conv2_out,
                                                     filters=self.image_size,
                                                     kernel_size=(5, 5),
                                                     strides=(2, 2),
                                                     padding="same",
                                                     kernel_initializer=tf.truncated_normal_initializer(
                                                         stddev=self.init_weight_stddev))
            batch_trans_conv3 = tf.layers.batch_normalization(inputs=trans_conv3,
                                                              momentum=self.momentum,
                                                              training=training,
                                                              epsilon=self.epsilon)
            trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3)

            # 32x32x64 -> 64x64x3
            logits = tf.layers.conv2d_transpose(inputs=trans_conv3_out,
                                                filters=3,
                                                kernel_size=(5, 5),
                                                strides=(2, 2),
                                                padding="same",
                                                kernel_initializer=tf.truncated_normal_initializer(
                                                    stddev=self.init_weight_stddev))
            out = tf.tanh(logits)
            return out

    def discriminator(self, images, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # convert original image to 32x32x64
            conv1 = tf.layers.conv2d(inputs=images,
                                     filters=self.image_size,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=self.init_weight_stddev))
            batch_norm1 = tf.layers.batch_normalization(inputs=conv1,
                                                        momentum=self.momentum,
                                                        epsilon=self.epsilon)
            conv1_out = tf.nn.leaky_relu(batch_norm1)

            # 32x32x64 -> 16x16x128
            conv2 = tf.layers.conv2d(inputs=conv1_out,
                                     filters=self.image_size * 2,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=self.init_weight_stddev))
            batch_norm2 = tf.layers.batch_normalization(inputs=conv2,
                                                        momentum=self.momentum,
                                                        epsilon=self.epsilon)
            conv2_out = tf.nn.leaky_relu(batch_norm2)

            # 16x16x128 -> 8x8x256
            conv3 = tf.layers.conv2d(inputs=conv2_out,
                                     filters=self.image_size * 4,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=self.init_weight_stddev))
            batch_norm3 = tf.layers.batch_normalization(inputs=conv3,
                                                        momentum=self.momentum,
                                                        epsilon=self.epsilon)
            conv3_out = tf.nn.leaky_relu(batch_norm3)

            # 8x8x256 -> 4x4x512
            conv4 = tf.layers.conv2d(inputs=conv3_out,
                                     filters=self.image_size * 8,
                                     kernel_size=(5, 5),
                                     strides=(2, 2),
                                     padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=self.init_weight_stddev))
            batch_norm4 = tf.layers.batch_normalization(inputs=conv4,
                                                        momentum=self.momentum,
                                                        epsilon=self.epsilon)
            conv4_out = tf.nn.leaky_relu(batch_norm4)

            flatten = tf.reshape(conv4_out, (-1, self.image_size * 8 * (self.image_size // 16) * (self.image_size // 16)))
            logits = tf.layers.dense(inputs=flatten,
                                     units=1,
                                     activation=None)
            output = tf.sigmoid(logits)
            return output, logits
