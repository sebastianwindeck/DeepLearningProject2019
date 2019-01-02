"""
This is a straightforward Python implementation of a generative adversarial network.
The code is drawn directly from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

A version of this model with explanatory notes is also available on GitHub
at https://github.com/jonbruner/generative-adversarial-networks.

This script requires TensorFlow and its dependencies in order to run. Please see
the readme for guidance on installing TensorFlow.

This script won't print summary statistics in the terminal during training;
track progress and see sample images in TensorBoard.
"""

import tensorflow as tf
import numpy as np
import datetime

import pickle
from six.moves import cPickle

from dataset import DataSet

def scale_on_x_list(x_list, scaler):
    """Scale list of ndarray. 
    """
    return [scaler.transform(e) for e in x_list]
        

# Define the discriminator network
def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # First convolutional and pool layers
        # This finds 32 different 1 x 5 features
        d_w1 = tf.get_variable('d_w1', [1, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = d1 + d_b1			#   ?, 1, 88, 32
        d1 = tf.nn.relu(d1)		#   ?, 1, 88, 32
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')	# ?, 1, 44, 32

        # Second convolutional and pool layers
        # This finds 64 different 5 x 1 features
        d_w2 = tf.get_variable('d_w2', [1,  5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = d2 + d_b2		#  ?, 1, 44, 64
        d2 = tf.nn.relu(d2)	#  ?, 1, 44, 64
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')	# ?, 1, 22, 64

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [17600, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, 17600]) # ?, 17600
        d3 = tf.matmul(d3, d_w3)	#  ?, 1024
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)		#  ?, 1024

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4	# scalar

        # d4 contains unscaled values
        return d4

# Define the generator network
def generator(z, batch_size, z_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim, 44], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))  # 40, 44
    g_b1 = tf.get_variable('g_b1', [44], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1		#  ?, 44
    g1 = tf.reshape(g1, [-1, 88,  1, 1])	#  ?, 88, 1, 1
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.relu(g1)				#  ?, 88, 1, 1

    # Generate 88 features		should change to 2 * 88 features	TODO 
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)) # 3, 3, 1, 20
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 1, 1], padding='SAME')		# ?, 44, 1, 20
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')		# ?, 44, 1, 20
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [1, 88])		# ?, 1, 88, 20

    # Generate 88 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)) # 3, 3, 20, 10
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')	# ?, 1, 44, 10
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')	# ?, 1, 44, 10
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [1, 88])	# ?, 1, 88, 10

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))  # 1, 1, 10, 1
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 1, 1, 1], padding='SAME')	#  ?, 1, 88, 1
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)			#	?, 1, 88, 1		

    # Dimensions of g4: batch_size x  1 x 88 x 1	
    return g4

if __name__ == '__main__':

    z_dimensions = 40	# z_placeholder is input to the generator
    batch_size = 50
    index_in_epoch = 0
    tr_packed_feat_path = "./packed_features/logmel/train.p" 	#     MODIFY as needed TODO
    scaler_path = "./scalers/logmel/scaler.p"			#     MODIFY as needed TODO
    
    [tr_x_list, tr_y_list, tr_na_list] = cPickle.load(open(tr_packed_feat_path, 'rb'))
    #[te_x_list, te_y_list, te_na_list] = cPickle.load(open(te_packed_feat_path, 'rb'))
    
    # Scale. 
    if True:
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x_list = scale_on_x_list(tr_x_list, scaler)
    
    y_all = []
    for i in range(len(tr_y_list)):
        y_all.append(tr_y_list[i])
    y_all = np.asarray(y_all)
    y_all =  np.concatenate(y_all, axis = 0)
    
    ds = DataSet(y_all)

    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
    # z_placeholder is for feeding input noise to the generator
    
    x_placeholder = tf.placeholder(tf.float32, shape = [None, 1,88,1], name='x_placeholder')
    # x_placeholder is for feeding input images to the discriminator
    
    Gz = generator(z_placeholder, batch_size, z_dimensions)
    # Gz holds the generated images
    
    Dx = discriminator(x_placeholder)
    # Dx will hold discriminator prediction probabilities
    # for the real rolls        
    
    Dg = discriminator(Gz, reuse_variables=True)
    # Dg will hold discriminator prediction probabilities for generated rolls
    
    # Define losses
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))
    
    # Define variable lists
    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]
    
    # Define the optimizers
    # Train the discriminator
    d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)
    
    # Train the generator
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)
    
    # From this point forward, reuse variables
    tf.get_variable_scope().reuse_variables()
    
    sess = tf.Session()
    
    # Send summary statistics to TensorBoard
    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_loss_real', d_loss_real)
    tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
    
    rolls_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
    tf.summary.image('Generated_rolls', rolls_for_tensorboard, 5)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)
    
    sess.run(tf.global_variables_initializer())
    
    # Pre-train discriminator
    for i in range(100):
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        real_roll_batch = ds.get_next_batch(batch_size).reshape([batch_size, 1, -1, 1])
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                               {x_placeholder: real_roll_batch, z_placeholder: z_batch})
    
    # Train generator and discriminator together
    for i in range(1000):
        real_image_batch = ds.get_next_batch(batch_size).reshape([batch_size, 1, -1, 1])
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    
        # Train discriminator on both real and fake images
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                               {x_placeholder: real_image_batch, z_placeholder: z_batch})
    
        # Train generator
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})
    
        if i % 10 == 0:
            # Update TensorBoard with summary statistics
            z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
            summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_roll_batch})
            writer.add_summary(summary, i)

        if i % 20 == 0:
            print(i, "  ", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "   g_loss:", g_loss, "   d_loss_real:", d_loss_real, "    d_loss_fake:", d_loss_fake)		# TODO better resolution
    
    # Optionally, uncomment the following lines to update the checkpoint files attached to the tutorial.
    # saver = tf.train.Saver()
    # saver.save(sess, 'pretrained-model/pretrained_gan.ckpt')
