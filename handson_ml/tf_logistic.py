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


### tensor flow
def logistic_regression(X_train, y_train, n_epochs=1000, lrate=0.01):
    m, n = X_train.shape
    X = tf.placeholder(tf.float32, shape=(m, n), name='X')
    y = tf.placeholder(tf.float32, shape=(m, 1), name='y')
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
            if epoch % 100 == 0:
                print("Epoch", epoch, "Los = ", sess.run(loss, {X:X_train, y:y_train.reshape(m,1)}))
                      #"Prob = ", sess.run(prob, {X: X_tf, y: y_tf}),
                      #"Score =", sess.run(score, {X: X_tf, y: y_tf}),
                      #"W = ", w_tf.eval(), "c=", c_tf.eval())
            sess.run(train_op, {X:X_train, y:y_train.reshape(m,1)})
        
        best_w = w.eval()
        best_c = c.eval()
    return best_w, best_c

w0, c0 = logistic_regression(X_sk, y_sk)



n_epochs = 3000
X_tf = X_sk
y_tf = y_sk.reshape(-1,1)


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

w_tf = tf.Variable(tf.random_uniform([n, 1], -1., 1.), dtype=tf.float32, name='W')
c_tf = tf.Variable(tf.random_uniform([1, 1], -1., 1.), dtype=tf.float32, name='c')

score = tf.multiply(tf.matmul(X, w_tf) + c_tf, y)
prob = tf.sigmoid(score)
loss = 0.5 * tf.reduce_sum(w_tf * w_tf)- C* tf.reduce_sum(tf.log(prob)) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.005)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "Los = ", sess.run(loss, {X:X_tf, y:y_tf}))
                  #"Prob = ", sess.run(prob, {X: X_tf, y: y_tf}),
                  #"Score =", sess.run(score, {X: X_tf, y: y_tf}),
                  #"W = ", w_tf.eval(), "c=", c_tf.eval())
        sess.run(train_op, {X:X_tf, y:y_tf})
        
    best_w = w_tf.eval()
    best_c = c_tf.eval()

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

