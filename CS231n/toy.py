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
