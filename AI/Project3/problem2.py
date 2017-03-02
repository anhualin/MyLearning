# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:56:17 2017

@author: alin
"""

import numpy as np
from sklearn import preprocessing
import sys

def scaleFeature(data):
    X = data[:, 0:2]
    y = data[:,2]
    X_scaled = preprocessing.scale(X)
    one = np.ones([X.shape[0], 1])
    X_new = np.concatenate((one, X_scaled), axis = 1)
    return X_new, y

def gradientDescent(X, y, f):
    N, m = X.shape
    y = y.reshape([N,1])
    alphas = [.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 0.02]
    for alpha in alphas:
        beta = np.zeros([m,1])
        for i in range(100):
            beta = beta -alpha/N * X.T.dot(X.dot(beta) - y)
        f.write(str(alpha) + ',100,' + str(beta[0,0]) + ',' + str(beta[1,0]) + ',' + str(beta[2,0]) + '\n')

def main(argv):
    input_file = argv[0]
#    input_file = 'input2.csv' 
    data = np.loadtxt(input_file, delimiter = ',')
    output = argv[1]
    f = open(output, 'w')

    X, y = scaleFeature(data)
    gradientDescent(X, y, f)
if __name__ == "__main__":
    main(sys.argv[1:])