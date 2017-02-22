# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:56:17 2017

@author: alin
"""

import numpy as np
import sys

def hw2_train(X_train, y_train):
    categories = np.unique(y_train)
    categories = list(set(y_train))
    categories.sort()
    N = X_train.shape[0]
    cats = {}
    inds = range(N)
    for c in categories:
        g = [e == c for e in y_train]
        cats[c] = [e for e in inds if g[e]]

    dists = []
    for c in categories:
        pi_c = len(cats[c])*1.0 / N
        X_c = X_train[cats[c],]
        mu_c = X_c.mean(axis = 0)
        X_cc = X_c - mu_c
        Sigma_c = X_cc.T.dot(X_cc)/len(cats[c])
        det = 1/np.sqrt(np.linalg.det(Sigma_c))
        dist = {'pi': pi_c, 'det': det, 'mu': mu_c, 'Sigma': Sigma_c}
        dists.append(dist)
    return categories, dists

def hw2_classify(categories, dists, X_test):
    result = []
    for i in range(X_test.shape[0]):
        f = []
        x = X_test[i,].T
        for c in categories:
            dist = dists[c]
            xmu = x-dist['mu']
            b = np.linalg.solve(dist['Sigma'], xmu)
            d = -0.5 * xmu.T.dot(b)
            fc = dist['pi'] * dist['det']*np.exp(d)
            f.append(fc)
        prob = [e/sum(f) for e in f]
        result.append(prob)
    return result

def load_data(X_train_file, y_train_file, X_test_file):
    X_train = np.loadtxt(X_train_file, delimiter = ',')
    y_train = np.loadtxt(y_train_file, delimiter = ',').tolist()
    y_train =  [int(e) for e in y_train]
    X_test = np.loadtxt(X_test_file, delimiter = ',' )
    return X_train, y_train, X_test

def write_output(result):

    file_name = 'probs_test.csv'
    f = open(file_name, 'w')

    for r in result:
        rs = ','.join([str('{0:.10f}'.format(e)) for e in r])
        f.write(str(rs) + '\n')
    f.close()

def main(argv):
    X_train_file = argv[0]
    y_train_file = argv[1]
    X_test_file = argv[2]
    X_train, y_train, X_test = load_data(X_train_file, y_train_file, X_test_file)

    categories, dists = hw2_train(X_train, y_train)
    result = hw2_classify(categories, dists, X_test)
    write_output(result)

if __name__ == "__main__":
    main(sys.argv[1:])