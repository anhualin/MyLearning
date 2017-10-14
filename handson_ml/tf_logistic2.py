#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 20:58:43 2017

@author: alin
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LinearRegression
lgr = LinearRegression()
lgr.fit(X_train, y_train)
y_pred_prob = lgr.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, np.ones(y_test.shape[0])))

m_train, n_train = X_train.shape
X_train_tf = np.c_[np.ones((m_train,1)), X_train]
y_train_tf = y_train.reshape(m_train,1)
m_test, n_test = X_test.shape
X_test_tf = np.c_[np.ones((m_test, 1)), X_test]
y_test_tf = y_test.reshape(m_test,1)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def logistic_regression(X_train, y_train, n_epochs=1000, lrate=0.01, batch_size=50):
    reset_graph()
    m, n = X_train.shape
    n_batches = int(np.ceil(m / batch_size))
  
    X = tf.placeholder(tf.float32, shape=(None, n), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    w = tf.Variable(tf.random_uniform([n, 1], -1.0, 1.0, seed=42), name='w')
    score = tf.matmul(X, w)
    prob = tf.sigmoid(score)
    loss = tf.losses.log_loss(y, prob)
    epsilon = 1e-7  # to avoid an overflow when computing the log
    loss1 = -tf.reduce_mean(y * tf.log(prob + epsilon) + (1 - y) * tf.log(1 - prob + epsilon))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
    train_op = optimizer.minimize(loss)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                sess.run(train_op, feed_dict={X: X_train, y:y_train})
            if epoch %100 == 0:
                loss_val = loss.eval({X: X_train, y:y_train})
                loss_val1 = loss1.eval({X: X_train, y:y_train})
                print("Epoch:", epoch, "\tLoss:", loss_val, "\tLoss1:", loss_val1)
        best_w = w.eval()
        return best_w
w_tf = logistic_regression(X_train_tf, y_train_tf)

def log_prob(X,w):
    score = np.dot(X,w)
    prob = 1/(1 + np.exp(-score))
    return prob
y_prob_tf = log_prob(X_test_tf, w_tf)
y_pred_tf = (y_prob_tf > 0.5).astype(int)
print(accuracy_score(y_test, y_pred_tf))


from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=3)
X_train_tf2 = pf.fit_transform(X_train)
X_test_tf2 = pf.transform(X_test)

w_tf2 = logistic_regression(X_train_tf2, y_train_tf)
y_prob_tf2 = log_prob(X_test_tf2, w_tf2)
y_pred_tf2 = (y_prob_tf2 > 0.5).astype(int)
print(accuracy_score(y_test, y_pred_tf2))
