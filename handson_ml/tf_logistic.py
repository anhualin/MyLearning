# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:00:36 2017

@author: alin
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

C = 3.0
t = 20
m = 2*t
n = 8
X_sk = np.random.rand(m,n)
y_sk = np.r_[np.ones(t), -np.ones(t)]


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
print(np.linalg.norm(grad_sk))

#
#### tensor flow
#def fetch_batch(epoch, batch_index, batch_size):
#    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
#    indices = np.random.randint(m, size=batch_size)  # not shown
#    X_batch = scaled_housing_data_plus_bias[indices] # not shown
#    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
#    return X_batch, y_batch

def fetch_batch_indices(epoch, batch_index, batch_size, m):
    np.random.seed(epoch + batch_index)
    #indices = np.random.randint(m, size=batch_size)
    indices = np.random.choice(m, size=batch_size, replace=False)
    return indices

def logistic_regression(X_train, y_train, n_epochs=1000, lrate=0.01, batch_size=10):
    m, n = X_train.shape
    #n_batches = int(np.ceil(m / batch_size))
    n_batches = 1
    batch_size = m
    y_train = y_train.reshape(-1, 1)
    X = tf.placeholder(tf.float32, shape=(None, n), name='X')
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    w = tf.Variable(tf.random_uniform([n, 1], -1., 1.), dtype=tf.float32, name='W')
    c = tf.Variable(tf.random_uniform([1, 1], -1., 1.), dtype=tf.float32, name='c')

    score = tf.multiply(tf.matmul(X, w) + c, y)
    prob = tf.sigmoid(score)
    loss = 0.5 * tf.reduce_sum(w * w)- C* tf.reduce_sum(tf.log(prob)) 

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
    train_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                batch_indices = fetch_batch_indices(epoch, batch_index, batch_size, m)
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                #X_batch = X_train
                #y_batch = y_train
                sess.run(train_op, feed_dict={X:X_batch, y:y_batch})
        
        best_w = w.eval()
        best_c = c.eval()
    return best_w, best_c

w0, c0 = logistic_regression(X_train=X_sk, y_train=y_sk, n_epochs=1000, batch_size=400)



w_tf = w0
c_tf = c0
score_tf = np.dot(X_sk, w_tf) + c_tf
score_tf = score_tf.reshape(m,)
loss_tf = 0.5 * np.sum(w_tf * w_tf) +  C * np.sum(np.log(np.exp(-y_sk * score_tf)+1))

coef_tf = y_sk/(1 + np.exp(y_sk* score_tf))
coef_tf = coef_tf.reshape(m,1)
grad_tf = w_tf - C * np.dot(X_sk.T, coef_tf)
print(np.linalg.norm(grad_tf))


    
score_sk = np.dot(X_sk, w_sk) + c_sk
score_sk = score_sk.reshape(m,)
loss_sk = 0.5 * np.sum(w_sk * w_sk) +  C * np.sum(np.log(np.exp(-y_sk * score_sk)+1))

coef_sk = y_sk/(1 + np.exp(y_sk* score_sk))
coef_sk = coef_sk.reshape(m,1)
grad_sk = w_sk - C * np.dot(X_sk.T, coef_sk)
print(np.linalg.norm(grad_sk))

