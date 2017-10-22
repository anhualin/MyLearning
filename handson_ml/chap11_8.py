#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 22:02:14 2017

@author: alin
"""

from sys import platform
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

if platform == 'win32':
    tmpdir = 'C:/Users/alin/Documents/Temp/data'
else:
    tmpdir = '/tmp/data/'
    
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
    
n_inputs = 28 * 28  # MNIST
n_hidden1 = 100
n_hidden2 = 100
n_hidden3 = 100
n_hidden4 = 100
n_hidden5 = 100
n_outputs = 5

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

with tf.name_scope('dnn'):
    he_init = tf.contrib.layers.variance_scaling_initializer()
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, 
                              kernel_initializer=he_init, name='hidden1')
    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu, 
                              kernel_initializer=he_init, name='hidden2')
    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.elu, 
                              kernel_initializer=he_init, name='hidden3')
    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.elu, 
                              kernel_initializer=he_init, name='hidden4')
    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.elu, 
                              kernel_initializer=he_init, name='hidden5')
    logits = tf.layers.dense(hidden5, n_outputs, kernel_initializer=he_init, name='output')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
        
learning_rate = 0.01
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
n_epochs = 20
batch_size = 200
n_batchs = X_train.shape[0] // batch_size
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batchs):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch1 = X_batch[y_batch < 5]
            y_batch1 = y_batch[y_batch < 5]
            sess.run(training_op, feed_dict={X: X_batch1, y: y_batch1})
        accuracy_val = accuracy.eval({X: X_valid1, y:y_valid1})
        print(epoch, '  Valid accuracy:',  accuracy_val)
    
    
    
