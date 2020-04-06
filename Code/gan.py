import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from utils import sample_noise, preprocess_img, show_images
from loss import generator_loss,discriminator_loss
from data_helper import MNIST


def gan_run(D, G, num_epochs,learning_rate, beta1, print_every, batch_size, noise_size):


    mnist = MNIST(batch_size=batch_size, shuffle=True)
    D_solver = tf.keras.optimizers.Adam(learning_rate=1e-3,beta_1=0.5)
    G_solver = tf.keras.optimizers.Adam(learning_rate=1e-3,beta_1=0.5)

    iter_count = 0
    for epoch in range(num_epochs):
        for (x, _) in mnist:
            with tf.GradientTape() as tape:
                real_data = x
                logits_real = D(preprocess_img(real_data))

                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)
                logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))

                d_total_error = discriminator_loss(logits_real, logits_fake)
                d_gradients = tape.gradient(d_total_error, D.trainable_variables)
                D_solver.apply_gradients(zip(d_gradients, D.trainable_variables))

            with tf.GradientTape() as tape:
                g_fake_seed = sample_noise(batch_size, noise_size)
                fake_images = G(g_fake_seed)

                gen_logits_fake = D(tf.reshape(fake_images, [batch_size, 784]))
                g_error = generator_loss(gen_logits_fake)
                g_gradients = tape.gradient(g_error, G.trainable_variables)
                G_solver.apply_gradients(zip(g_gradients, G.trainable_variables))

            if (iter_count % print_every == 0):
                print('Epoch: {}, Iter: {}, D: {:.4}, G:{:.4}'.format(epoch, iter_count,d_total_error,g_error))
        
            iter_count += 1

    z = sample_noise(batch_size, noise_size)
    G_sample = G(z)
    print('Final images')
    show_images(G_sample[:16])
    plt.show()
