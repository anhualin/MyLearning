# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:00:36 2017

@author: alin
"""

import numpy as np
import tensorflow as tf

n_epochs = 100
X_train = np.random.rand(3,4)
m, n = X_train.shape
y_train = np.random.randint(0, 2, m).reshape(-1,1)


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

w = tf.Variable(tf.random_uniform([n, 1], -1., 1.), dtype=tf.float32, name='W')
b = tf.Variable(tf.random_uniform([1, 1], -1., 1.), dtype=tf.float32, name='b')

score = tf.matmul(X, w) + b
prob = tf.sigmoid(score)
loss = -1/m * tf.reduce_sum(tf.multiply(y, tf.log(prob)) + tf.multiply(1-y, tf.log(1-prob)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        sess.run(train_op, {X:X_train, y:y_train})
        
    best_w = w.eval()