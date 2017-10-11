# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:00:36 2017

@author: alin
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

C = 2.0
t = 30
m = 2*t
n = 20
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




### tensor flow
n_epochs = 1000
X_tf = X_sk
y_tf = y_sk.reshape(-1,1)


tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n), name='X')
y = tf.placeholder(tf.float32, shape=(None, 1), name='y')

w_tf = tf.Variable(tf.random_uniform([n, 1], -1., 1.), dtype=tf.float32, name='W')
c_tf = tf.Variable(tf.random_uniform([1, 1], -1., 1.), dtype=tf.float32, name='c')

score = tf.matmul(X, w_tf) + c_tf
prob = tf.sigmoid(score)
loss = 0.5 * tf.tensordot(w_tf, w_tf, 1) + C* tf.reduce_sum() 

- tf.reduce_sum(tf.multiply(y, tf.log(prob)) + tf.multiply(1-y, tf.log(1-prob)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.05)
train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "Los = ", sess.run(loss, {X:X_train, y:y_train}),
                  "Prob = ", sess.run(prob, {X: X_train, y: y_train}),
                  "Score =", sess.run(score, {X: X_train, y: y_train}),
                  "W = ", w_tf.eval(), "c=", c_tf.eval())
        sess.run(train_op, {X:X_train, y:y_train})
        
    best_w = w_tf.eval()
    best_c = c_tf.eval()
    
w = -0.07532753
c = 0.01130767
g = x1 + x1/(1 + np.exp(w*x1 + c)) - x2/(1 + np.exp(w*x2 + c)) + w
g_raw = x1 + x1/(1 + np.exp(w_raw*x1 + c_raw)) - x2/(1 + np.exp(w_raw*x2 + c_raw)) + w_raw
a1 = w*x1 + c
b1 = w_raw*x1 + c_raw
a2 = w*x2 + c
b2 = w_raw*x2 + c_raw

a3 = np.exp(a1)
b3 = np.exp(b1)
a4 = np.exp(a2)
b4 = np.exp(b2)

