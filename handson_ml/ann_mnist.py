#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 14:43:55 2017

@author: alin
"""
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

X, y = mnist['data'], mnist['target']
y = y.astype(np.int)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

from sklearn.model_selection import train_test_split
X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist_tf = input_data.read_data_sets("/tmp/data/")

def ann_mnist1(X, y, hidden_units, n_classes):
    config = tf.contrib.learn.RunConfig(tf_random_seed=42)
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X)
    ann_clf = tf.contrib.learn.DNNClassifier(hidden_units=hidden_units,
                                             n_classes=n_classes,
                                             feature_columns=feature_columns,
                                             config=config)
    ann_clf = tf.contrib.learn.SKCompat(ann_clf)
    ann_clf.fit(X, y, batch_size=100,steps=40000)
    return ann_clf

def ann_select1(hidden_units, X_t, y_t, X_v, y_v):
    ann1 = ann_mnist1(X_t, y_t, hidden_units, 10)
    y_p = ann1.predict(X_v)['classes']
    print(accuracy_score(y_v, y_p))

ann_select1([300], X_t, y_t, X_v, y_v) #93.9%
ann_select1([300], X_t, y_t, X_t, y_t) #98.47
ann_select1([300,300], X_t, y_t, X_v, y_v) #92.97
ann_select1([300,300], X_t, y_t, X_t, y_t) #97.47%
ann_select1([300,100], X_t, y_t, X_v, y_v) #93.17
ann_select1([300,100], X_t, y_t, X_t, y_t) #97.94%

ann_select1([300], X_train, y_train, X_test, y_test) #94.9%
ann_select1([300, 300], X_train, y_train, X_test, y_test) #94.2%
ann_select1([300, 100], X_train, y_train, X_test, y_test) #94.37


#######################################

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    
def add_layer(X, n_out, name, activation=None):
    with tf.name_scope(name):
        n_in = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_in)
        init = tf.truncated_normal((n_in, n_out), stddev=stddev)
        W = tf.Variable(init, name='W')
        b = tf.Variable(tf.zeros([n_out]), name='b')
        z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(z)
        else:
            return z
        

    
def ann_mnist2(X_train, y_train, hidden_units, n_classes, lrate=0.01, n_epochs = 40,
               batch_size=50):
    reset_graph()
    m, n0 = X_train.shape
    n_batches = int(np.ceil(m / batch_size))
    X = tf.placeholder(tf.float32, shape=(None, n0), name='X')
    y = tf.placeholder(tf.int64, shape=(None), name='y')
    Z = X
    for i in range(len(hidden_units)):
        Z = add_layer(Z, hidden_units[i], 'level' + str(i), tf.nn.relu)
        
    logits = add_layer(Z, n_classes, 'output')
#    with tf.name_scope('dnn'):
#        Z1 = add_layer(X, hidden_units[0], 'level0', tf.nn.relu)
#        Z2 = add_layer(Z1, hidden_units[1], 'level1', tf.nn.relu)
#        logits = add_layer(Z2, n_classes, 'output')
    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
        training_op = optimizer.minimize(loss)
    
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = mnist_tf.train.next_batch(batch_size)
#                batch_indices = np.random.choice(m, batch_size, replace =False)
#                X_batch, y_batch = X_train[batch_indices], y_train[batch_indices] 
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if epoch % 5 == 0:
                loss_val = loss.eval({X: X_train, y:y_train})
                acc_train = accuracy.eval({X: X_train, y: y_train})
                print('Epoch:', epoch, ' Loss:', loss_val, 'Train_accuracy', acc_train)

ann_mnist2(X_train=X_t, y_train=y_t, hidden_units=[300,100], n_classes=10, 
           lrate=0.01, n_epochs = 20, batch_size=200)
ann_mnist2(X_train=X_train, y_train=y_train, hidden_units=[300,100], n_classes=10, 
           lrate=0.01, n_epochs = 10, batch_size=200)