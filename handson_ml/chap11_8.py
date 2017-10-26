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
from functools import partial


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

best_accuracy = {}

learning_rate = 0.005
batch_norm_momentum = 0.9
dropout_rate = 0.5
losses = []
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope('dnn'):
    
    he_init = tf.contrib.layers.variance_scaling_initializer()
    
    my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum = batch_norm_momentum
            )
    my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init)
    hidden1 = my_dense_layer(X, n_hidden1, name='hidden1')
   # hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1), name='bn1')
    hidden2 = my_dense_layer(bn1, n_hidden2, name='hidden2')
#    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2), name='bn2')
    hidden3 = my_dense_layer(bn2, n_hidden3, name='hidden3')
#    hidden3_drop = tf.layers.dropout(hidden3, dropout_rate, training=training)
    bn3 = tf.nn.elu(my_batch_norm_layer(hidden3), name='bn3')
    hidden4 = my_dense_layer(bn3, n_hidden4, name='hidden4')
#    hidden4_drop = tf.layers.dropout(hidden4, dropout_rate, training=training)
    bn4 = tf.nn.elu(my_batch_norm_layer(hidden4), name='bn4')
    hidden5 = my_dense_layer(bn4, n_hidden5, name='hidden5')
#    hidden5_drop = tf.layers.dropout(hidden5, dropout_rate, training=training)
    bn5 = tf.nn.elu(my_batch_norm_layer(hidden5), name='bn5')
    logits_before_bn = my_dense_layer(bn5, n_outputs, name='output')
    logits = my_batch_norm_layer(logits_before_bn)
#    
#    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, 
#                              kernel_initializer=he_init, name='hidden1')
#    hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.elu, 
#                              kernel_initializer=he_init, name='hidden2')
#    hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.elu, 
#                              kernel_initializer=he_init, name='hidden3')
#    hidden4 = tf.layers.dense(hidden3, n_hidden4, activation=tf.nn.elu, 
#                              kernel_initializer=he_init, name='hidden4')
#    hidden5 = tf.layers.dense(hidden4, n_hidden5, activation=tf.nn.elu, 
#                              kernel_initializer=he_init, name='hidden5')
#    logits = tf.layers.dense(hidden5, n_outputs, kernel_initializer=he_init, name='output')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
        

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()
n_epochs = 20
batch_size = 200
n_batchs = X_train.shape[0] // batch_size
best_accuracy[learning_rate] = (-np.inf, -np.inf)
no_improve = 0
threshold = 4
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        
        if no_improve < threshold:
            for batch_index in range(n_batchs):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                X_batch1 = X_batch[y_batch < 5]
                y_batch1 = y_batch[y_batch < 5]
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_batch1, y: y_batch1})
            losses.append(loss.eval({X: X_train1, y:y_train1}))
            accuracy_val = accuracy.eval({X: X_valid1, y:y_valid1})
            accuracy_train = accuracy.eval({X: X_train1, y:y_train1})
            print(epoch, 'Train accuracy:', accuracy_train,  '  Valid accuracy:',  accuracy_val)
            if accuracy_val > best_accuracy[learning_rate][1]:
                best_accuracy[learning_rate] = (accuracy_train, accuracy_val)
                save_path = saver.save(sess, "/tmp/chap11_8.ckpt")
                no_improve = 0
            else:
                no_improve += 1
with tf.Session() as sess:
    saver.restore(sess, "/tmp/chap11_8.ckpt")
    accuracy_test = accuracy.eval({X:X_test1, y:y_test1})
    print('test accuracy:', accuracy_test)
    
