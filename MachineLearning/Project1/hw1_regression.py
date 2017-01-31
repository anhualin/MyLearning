# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:56:17 2017

@author: alin
"""

import numpy as np
import sys
#
#lam = 0.1
#sigma2 = 1
#
#N = 100
#d = 4
#Xall = np.random.rand(N,d-1)
#Xall = np.append(Xall, np.ones([N,1]),1)
#err = np.random.rand(N,1)
#w = np.random.rand(d,1) * 10 - 5
#yall = Xall.dot(w) + err
#
#X = Xall[:60]
#y = yall[:60]
#
#X_new = Xall[60:]
#
#np.savetxt('X_train.csv', X, delimiter = ',')
#np.savetxt('y_train.csv', y)
#np.savetxt('X_test.csv', X_new, delimiter = ',')
#
#index = range(X_new.shape[0])
#y_new = yall[60:]
#
#X_train = X
#y_train = y

def hw_regression(lam, sigma2, X_train, y_train, X_test, top_num):
    d = X_train.shape[1]
    A = lam * np.eye(d) + X_train.T.dot(X_train)
    b = X_train.T.dot(y_train)
    w_RR = np.linalg.solve(A, b).reshape(d)

    
    invSigma = lam * np.eye(d) + (1/sigma2) * X_train.T.dot(X_train)
    active = []
    X_new = X_test
    index = range(X_new.shape[0])
    for i in range(top_num):
        g1 = np.linalg.solve(invSigma, X_new.T)
        t2 = X_new.dot(g1)
        sigma02 = np.diag(t2)
        choose = np.argmax(sigma02)
        active.append(index[choose])
        x0 = X_new[choose,].reshape([d,1])
        invSigma = invSigma + x0.dot(x0.T)
        X_new = np.delete(X_new, choose, 0)
        index.remove(index[choose])
    return w_RR, active

def load_data(X_train_file, y_train_file, X_test_file):
    X_train = np.loadtxt(X_train_file, delimiter = ',')
    y_train = np.loadtxt(y_train_file, delimiter = ',')
    y_train = y_train.reshape([y_train.shape[0], 1])
    X_test = np.loadtxt(X_test_file, delimiter = ',' )
    return X_train, y_train, X_test

def write_output(w_RR, active, lam, sigma2):
    file_name1 = 'w_RR_' + str(lam) + '.csv'
    file_name2 = 'active_' + str(lam) + '_'+str(sigma2) + '.csv'
    
    f = open(file_name1, 'w')
    for w in w_RR:
        f.write(str(w) + '\n')
    f.close()
    
    f = open(file_name2, 'w')
    for i in active[:-1]:
        f.write(str(i+1) + ',')
    f.write(str(active[-1]))
    f.close()
    
def main(argv):
    lam = float(argv[0])
    sigma2 = float(argv[1])
    X_train_file = argv[2]
    y_train_file = argv[3]
    X_test_file = argv[4]
    X_train, y_train, X_test = load_data(X_train_file, y_train_file, X_test_file)
    w_RR, active = hw_regression(lam, sigma2, X_train, y_train, X_test, 10)
   
    write_output(w_RR, active, lam, sigma2)
   
if __name__ == "__main__":
    main(sys.argv[1:])