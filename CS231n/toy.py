#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 23:27:25 2017

@author: alin
"""
import matplotlib.pyplot as plt
import numpy as np
N = 100
D = 2
K = 3

X = np.zeros(((N*K), D))
y = np.zeros(N*K, dtype = 'uint8')

for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
    
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()


# linear classifier
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1,K))

step_size = 1.0
reg = 1e-3

num_examples = X.shape[0]
for i in range(200):
    scores = np.dot(X, W) + b
    
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print("iteration %d: loss %f", (i, loss))
    
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis = 0, keepdims = True)
    
    dW += reg*W
    W += -step_size * dW
    b += -step_size * db
    
    
    
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis = 1)
print("training accuracty: %.2f" % (np.mean(predicted_class == y)))        


h = 100
W1 = 0.01 * np.random.randn(D, h)
b1 = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

step_size = 1.0
reg = 1e-3

num_examples = X.shape[0]
for i in range(20000):
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
    scores = np.dot(hidden_layer, W2) + b2
    
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W1*W1) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 1000 == 0:
        print("iteration %d: loss %f", (i, loss))

    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis = 0, keepdims = True)
    
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0
    
    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis = 0, keepdims = True)
    
    dW2 += reg * W2
    dW1 += reg * W1
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2
    
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2

predicted_class = np.argmax(scores, axis = 1)
print("training accuracy: %2.f" % (np.mean(predicted_class == y)))