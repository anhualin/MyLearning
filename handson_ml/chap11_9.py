#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:35:00 2017

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

X_train2_full = X_train[y_train >= 5]
y_train2_full = y_train[y_train >= 5] -5
X_valid2_full = X_valid[y_valid >= 5]
y_valid2_full = y_valid[y_valid >= 5] -5
X_test2 = X_test[y_test >= 5]
y_test2 = y_test[y_test >= 5] -5

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
he_init = tf.contrib.layers.variance_scaling_initializer()

reset_graph()

restore_saver = tf.train.import_meta_graph(modeldir + '/my_best_mnist_0_4.meta')
X = tf.get_default_graph().get_tensor_by_name('X:0')
y = tf.get_default_graph().get_tensor_by_name('y:0')
loss = tf.get_default_graph().get_tensor_by_name("loss:0")
Y_proba = tf.get_default_graph().get_tensor_by_name("Y_proba:0")
logits = Y_proba.op.inputs[0]
accuracy = tf.get_default_graph().get_tensor_by_name("accuracy:0")

learning_rate = 0.005
output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='logits')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam2')
training_op = optimizer.minimize(loss, var_list=output_layer_vars)


correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
init = tf.global_variables_initializer()
five_frozen_saver = tf.train.Saver()

##########################
def sample_n_instances_per_class(X, y, n=100):
    Xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        Xc = X[idx][:n]
        yc = y[idx][:n]
        Xs.append(Xc)
        ys.append(yc)
    return np.concatenate(Xs), np.concatenate(ys)

X_train2, y_train2 = sample_n_instances_per_class(X_train2_full, y_train2_full)
X_valid2, y_valid2 = sample_n_instances_per_class(X_valid2_full, y_valid2_full, n=30)

       
import time

n_epochs = 40
batch_size = 200

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty

with tf.Session() as sess:
    init.run()
    restore_saver.restore(sess, modeldir + '/my_best_mnist_0_4')
    for var in output_layer_vars:
        var.initializer.run()
    
    t0 = time.time()
    
    for epoch in range(n_epochs):
        rnd_idx = np.random.permutation(len(X_train2))
        for rnd_indicies in np.array_split(rnd_idx, len(X_train2) // batch_size):
            X_batch, y_batch = X_train2[rnd_indicies], y_train2[rnd_indicies]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        
        loss_val, acc_val = sess.run([loss,accuracy], feed_dict={X: X_valid2, y: y_valid2})
        if loss_val < best_loss:
            save_path = five_frozen_saver.save(sess, modeldir + '/my_mnist_5_9_frozen')
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print ('Early stopping!')
                break
        print("{}\tValidation loss: {:.6f}\tBest loss: {:.6f}\tAccuracy: {:.2f}%".format(epoch, loss_val, best_loss, acc_val * 100))

t1 = time.time()
print("Total training time: {:.1f}s".format(t1 - t0))

with tf.Session() as sess:
    five_frozen_saver.restore(sess, modeldir + '/my_mnist_5_9_frozen')
    acc_test = accuracy.eval(feed_dict={X: X_test2, y: y_test2})
    print('test accuracy:', acc_test)