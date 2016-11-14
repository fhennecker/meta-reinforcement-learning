# autoencoder heavily inspired from https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10

nh1 = 256
nh2 = 128
n_input = 784

X = tf.placeholder("float", [None, n_input])

We1 = tf.Variable(tf.random_normal([n_input, nh1]))
We2 = tf.Variable(tf.random_normal([nh1, nh2]))
Wd1 = tf.Variable(tf.random_normal([nh2, nh1]))
Wd2 = tf.Variable(tf.random_normal([nh1, n_input]))

be1 = tf.Variable(tf.random_normal([nh1]))
be2 = tf.Variable(tf.random_normal([nh2]))
bd1 = tf.Variable(tf.random_normal([nh1]))
bd2 = tf.Variable(tf.random_normal([n_input]))


def encoder(x):
    layer1 = tf.sigmoid(tf.matmul(x, We1) + be1)
    layer2 = tf.sigmoid(tf.matmul(layer1, We2) + be2)
    return layer2

def decoder(x):
    layer1 = tf.sigmoid(tf.matmul(x, Wd1) + bd1)
    layer2 = tf.sigmoid(tf.matmul(layer1, Wd2) + bd2)
    return layer2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true-y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)

    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={X:batch_xs})

        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost:", "{:.9f}".format(c))

    print("Done.")
    encode_decode = sess.run(
            y_pred, feed_dict={X:mnist.test.images[:examples_to_show]})
    
    f, a = plt.subplots(2, 10, figsize=(10,2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28,28)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()

