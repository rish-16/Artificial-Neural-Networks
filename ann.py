import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_hl1 = 500
n_hl2 = 500
n_hl3 = 500
n_hl4 = 500
n_hl5 = 500
n_hl6 = 500

n_classes = 10
batch_size = 50

X = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float')

def ANN_model(data):

	# Training Weights
	w1 = tf.Variable(tf.random_normal([784, n_hl1]))
	b1 = tf.Variable(tf.random_normal([n_hl1]))

	w2 = tf.Variable(tf.random_normal([n_hl1, n_hl2]))
	b2 = tf.Variable(tf.random_normal([n_hl2]))

	w3 = tf.Variable(tf.random_normal([n_hl2, n_hl3]))
	b3 = tf.Variable(tf.random_normal([n_hl3]))

	w4 = tf.Variable(tf.random_normal([n_hl3, n_hl4]))
	b4 = tf.Variable(tf.random_normal([n_hl4]))

	w5 = tf.Variable(tf.random_normal([n_hl4, n_hl5]))
	b5 = tf.Variable(tf.random_normal([n_hl4]))

	w6 = tf.Variable(tf.random_normal([n_hl5, n_hl6]))
	b6 = tf.Variable(tf.random_normal([n_hl4]))

	w_out = tf.Variable(tf.random_normal([n_hl6, n_classes]))
	b_out = tf.Variable(tf.random_normal([n_classes]))

	# 6 Hidden Layers
	l1 = tf.add(tf.matmul(data, w1), b1)
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, w2), b2)
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, w3), b3)
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3, w4), b4)
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4, w5), b5)
	l5 = tf.nn.relu(l5)

	l6 = tf.add(tf.matmul(l5, w6), b6)
	l6 = tf.nn.relu(l6)

	output = tf.matmul(l6, w_out) + b_out

	return output

def train_ANN(x):
	prediction = ANN_model(x)
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

train_ANN(X)
