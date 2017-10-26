# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:29:42 2017

@author: alin
"""

from sys import platform
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from functools import partial


if platform == 'win32':
    tmpdir = 'C:/Users/alin/Documents/Temp/data'
    modeldir = 'C:/Users/alin/Documents/Temp/model'
else:
    tmpdir = '/tmp/data'
    modeldir = '/tmp/model'
    
    
mnist = input_data.read_data_sets(tmpdir)
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype('int')
y_test = mnist.test.labels.astype('int')
X_valid = mnist.validation.images
y_valid = mnist.validation.labels.astype('int')

X_train1 = X_train[y_train < 5]
y_train1 = y_train[y_train < 5]
X_test1 = X_test[y_test < 5]
y_test1 = y_test[y_test < 5]
X_valid1 = X_valid[y_valid < 5]
y_valid1 = y_valid[y_valid < 5]

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
he_init = tf.contrib.layers.variance_scaling_initializer()

def dnn(inputs, n_hidden_layers=5, n_neurons=100, name=None, activation=tf.nn.elu,
        initializer=he_init):
    with tf.variable_scope('dnn'):
        for layer in range(n_hidden_layers):
            inputs = tf.layers.dense(inputs, n_neurons, activation=activation,
                                     kernel_initializer=initializer,
                                     name='hidden%d' %(layer + 1))
        return inputs
    
n_inputs = 28 * 28
n_outputs = 5
learning_rate = 0.005
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

dnn_outputs = dnn(X)
logits = tf.layers.dense(dnn_outputs, n_outputs, kernel_initializer=he_init, name='logits')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
        

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, name='training_op')
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 100
batch_size = 50
max_checks_without_success = 20
checks_without_success = 0
best_loss = np.inf

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(X_train1.shape[0])
        for rnd_indicies in np.array_split(rnd_idx, len(X_train1) // batch_size):
            X_batch, y_batch = X_train1[rnd_indicies], y_train1[rnd_indicies]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val, acc_val = sess.run([loss, accuracy], feed_dict={X: X_valid1, y: y_valid1})
        loss_train, acc_train = sess.run([loss, accuracy], feed_dict={X: X_train1, y: y_train1})
        if loss_val < best_loss:
            best_loss = loss_val
            save_path = saver.save(sess, modeldir + '/my_mnist_model_0_4.ckpt')
            checks_without_success = 0
        else:
            checks_without_success += 1
            if checks_without_success > max_checks_without_success:
                print('Early stopping')
                break
        print('Epoch:', epoch, '  Train loss:', loss_train, '  Valid loss:', loss_val)
        print('Epoch:', epoch, '  Train acc:', acc_train, '   Valid acc:', acc_val)

with tf.Session() as sess:
    saver.restore(sess, modeldir + '/my_mnist_model_0_4.ckpt')
    acc_test = sess.run(accuracy, feed_dict={X: X_test1, y: y_test1})
    print('Test accuracy: ', acc_test)