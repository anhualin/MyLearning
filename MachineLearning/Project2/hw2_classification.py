# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:56:17 2017

@author: alin
"""

import numpy as np
import sys
d = 3
N = 100
X_train = np.random.rand(N, d)

y_train = np.random.randint(0, 10, N)
categories = np.unique(y_train)
cats = {}
inds = range(N)
for c in categories:
    g = y_train == c
    cats[c] = [e for e in inds if g[e]]

dists = []
for c in categories:
    pi_c = len(cats[c])*1.0 / N
    X_c = X_train[cats[c],]
    mu_c = X_c.sum(axis = 0)/len(cats[c])
    X_cc = X_c - mu_c
    Sigma_c = X_cc.T.dot(X_cc)/len(cats[c])
    det = 1/np.sqrt(np.linalg.det(Sigma_c))
    dist = {'pi': pi_c, 'det': det, 'mu': mu_c, 'Sigma': Sigma_c}
    dists.append(dist)

X_test = np.random.rand(20, d)
def get_prob(dists, X_test):
    out = []
    for i in range(X_test.shape[0]):
        f = range(len(dists))
        x = X_test[i,].T
        dist = dists[i]
        xmu = x-dist['mu']
        b = np.linalg.solve(dist['Sigma'], xmu)
        d = -0.5 * xmu.T.dot(xmu)
        fi = dist['pi'] * dist['det']*np.exp(d)
        f.append(fi)
    prob = [e/sum(f) for e in f]
    out.append(prob)
    return out

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

#
#invE = 2 * np.eye(3) + (1/3.0) * X_train.T.dot(X_train)
#E = np.linalg.inv(invE)
#D = (X_test.dot(E)).dot(X_test.T)
#s = np.diag(D)
#choose = np.argmax(s)
#
#invSigma = 2.0 * np.eye(3) + (1/3.0) * X_train.T.dot(X_train)
#index = range(X_test.shape[0])
#g1 = np.linalg.solve(invSigma, X_test.T)
#t2 = X_test.dot(g1)
#sigma02 = np.diag(t2)
#choose = np.argmax(sigma02)
#active.append(index[choose]+1)
        
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
        active.append(index[choose]+1)
        x0 = X_new[choose,].reshape([d,1])
        invSigma = invSigma + x0.dot(x0.T)
        #Sigma = np.linalg.inv(invSigma)
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
        f.write(str(i) + ',')
    f.write(str(active[-1]))
    f.close()

# X_train, y_train, X_test = load_data('X_train.csv', 'y_train.csv', 'X_test.csv')
    
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