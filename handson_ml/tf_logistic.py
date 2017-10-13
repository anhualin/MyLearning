# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:00:36 2017

@author: alin
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from sklearn.datasets import make_moons
X_moons, y_moons = make_moons(n_samples=1000, noise=0.1, random_state=42)

y_sk = 2 * y_moons - 1
X_sk = X_moons
m, n = X_sk.shape
C = 3.0
X_tf1 = X_sk
y_tf1 = y_sk


lgr = LogisticRegression(random_state=42, tol=1e-6, C=C)
lgr.fit(X_sk, y_sk)
w_sk = lgr.coef_.reshape(n,1)
c_sk = lgr.intercept_
score_sk = np.dot(X_sk, w_sk) + c_sk
score_sk = score_sk.reshape(m,)
loss_sk = 0.5 * np.sum(w_sk * w_sk) +  C * np.sum(np.log(np.exp(-y_sk * score_sk)+1))
coef_sk = y_sk/(1 + np.exp(y_sk* score_sk))
coef_sk = coef_sk.reshape(m,1)
grad_sk = w_sk - C * np.dot(X_sk.T, coef_sk)

def grad1(C, w, c, X, y):
    m, n = X.shape
    score = np.dot(X, w) + c
    score = score.reshape(m, )
    coef = y/(1 + np.exp(y * score))
    coef = coef.reshape(m, 1)
    if C > 100:
        grad = - np.dot(X.T, coef)
    else:
        grad = w - C * np.dot(X.T, coef)
    return np.linalg.norm(grad)

print(grad1(C, w_sk, c_sk, X_sk, y_sk))

#
#### tensor flow1

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def logistic_regression1(X_train, y_train, C=1.0, n_epochs=1000, lrate=0.01, batch_size=100):
    reset_graph()
    m, n = X_train.shape
    n_batches = int(np.ceil(m / batch_size))
    y_train = y_train.reshape(-1, 1)
    X = tf.placeholder(tf.float32, shape=(None, n), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    w = tf.Variable(tf.random_uniform([n, 1], -1., 1.), dtype=tf.float32, name='W')
    c = tf.Variable(tf.random_uniform([1, 1], -1., 1.), dtype=tf.float32, name='c')

    score = tf.multiply(tf.matmul(X, w) + c, y)
    prob = tf.sigmoid(score)
    if C > 100: 
        alpha = 0
    else:
        alpha = 1/C
    loss = alpha* 0.5 * tf.reduce_sum(w * w) - tf.reduce_mean(tf.log(prob + 1e-7)) 

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)               
                sess.run(train_op, feed_dict={X:X_batch, y:y_batch})
            if epoch % 100 == 0:
                print("Epoch:", epoch, "\tLoss:", loss.eval({X: X_train, y:y_train}))
            
        best_w = w.eval()
        best_c = c.eval()
    return best_w, best_c

w_tf1, c_tf1 = logistic_regression1(X_train=X_tf1, y_train=y_tf1, C=1000,
                             n_epochs=1000, lrate = 0.01, batch_size=50)

print(grad1(C, w_tf1, c_tf1, X_tf1, y_tf1))

X_tf2 = np.c_[np.ones((m,1)), X_tf1]
y_tf2 = y_moons.reshape(m,1)

def logistic_regression2(X_train, y_train, n_epochs=1000, lrate=0.01, batch_size=50):
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
w_tf2 = logistic_regression2(X_tf2, y_tf2)


X_tf3 = X_tf1
y_tf3 = y_tf2
def logistic_regression3(X_train, y_train, n_epochs=1000, lrate=0.01, batch_size=50):
    reset_graph()
    m, n = X_train.shape
    n_batches = int(np.ceil(m / batch_size))
    y_train = y_train.reshape(-1, 1)
    X = tf.placeholder(tf.float32, shape=(None, n), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    w = tf.Variable(tf.random_uniform([n, 1], -1., 1.), dtype=tf.float32, name='W')
    c = tf.Variable(tf.random_uniform([1, 1], -1., 1.), dtype=tf.float32, name='c')

    score = tf.matmul(X, w) + c
    prob = tf.sigmoid(score)
    #loss = -tf.reduce_sum(y *  tf.log(prob + 1e-7) + (1-y) * tf.log(1 - prob + 1e-7))/m
    loss = -tf.reduce_mean(y *  tf.log(prob + 1e-7) + (1-y) * tf.log(1 - prob + 1e-7))
   # loss1 = tf.losses.log_loss(y, prob)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)               
                sess.run(train_op, feed_dict={X:X_batch, y:y_batch})
            if epoch % 100 == 0:
                print("Epoch:", epoch, "\tLoss:", loss.eval({X: X_train, y:y_train}))
            
        best_w = w.eval()
        best_c = c.eval()
    return best_w, best_c

w_tf3, c_tf3 = logistic_regression3(X_train=X_tf3, y_train=y_tf3)