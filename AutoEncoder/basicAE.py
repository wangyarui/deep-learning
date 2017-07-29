# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

print('TensorFlow version: %s' % tf.__version__)

import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0, one_hot=False)

img = mnist.train.images[20]
plt.imshow(img.reshape((28,28)))
#plt.axis('off')
#plt.show()



hidden_units = 64
input_units = mnist.train.images.shape[1]

inputs_ = tf.placeholder(tf.float32, (None, input_units), name='inputs_')
targets_ = tf.placeholder(tf.float32, (None, input_units), name='targets_')

hidden_ = tf.layers.dense(inputs_, hidden_units, activation=tf.nn.relu)

logits_ = tf.layers.dense(hidden_, input_units, activation=None)
outputs_ = tf.sigmoid(logits_, name='outputs_')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits_)
cost = tf.reduce_mean(loss)

learning_rate = 0.01
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()

epochs = 20
batch_size = 128
sess.run(tf.global_variables_initializer())

for e in range(epochs):
    for idx in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size) # 获取下一个batch
        batch_cost, _ = sess.run([cost, optimizer], 
                                 feed_dict={inputs_: batch[0],
                                            targets_: batch[0]})
        print("Epoch: {}/{}".format(e+1, epochs), 
              "Training loss: {:.4f}".format(batch_cost))


fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(20,8))
test_imgs = mnist.test.images[:5]
reconstructed, compressed = sess.run([outputs_, hidden_],
                                     feed_dict={inputs_: test_imgs})

for image, row in zip([test_imgs, reconstructed], axes):
    for img, ax in zip(image, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.show()
fig.tight_layout(pad=0.1)






