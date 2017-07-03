import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
batch_size = 128

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def CNN_model(x):

	# Training Weights
	w_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
	w_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
	w_fc = tf.Variable(tf.random_normal([7*7*64, 1024]))
	w_out = tf.Variable(tf.random_normal([1024, n_classes]))

	b_conv1 = tf.Variable(tf.random_normal([32]))
	b_conv2 = tf.Variable(tf.random_normal([64]))
	b_fc = tf.Variable(tf.random_normal([1024]))
	b_out = tf.Variable(tf.random_normal([n_classes]))

	x = tf.reshape(x, shape=[-1,28,28,1])

	conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
	conv1 = max_pool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, w_conv2) + b_conv2)
	conv2 = max_pool2d(conv2)

	fc = tf.reshape(conv2, [-1,7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, w_fc) + b_fc)

	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, w_out) + b_out

	return output

def train_CNN(x):
	prediction = CNN_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={X: epoch_x, Y: epoch_y})

				epoch_loss += c
			print ('Epoch', epoch+1, 'Loss: ', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(Y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))*100

		print('Accuracy: ', accuracy.eval({X:mnist.test.images, Y:mnist.test.labels}))

train_CNN(X)
