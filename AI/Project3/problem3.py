# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:56:17 2017

@author: alin
"""

import numpy as np
##from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#import sys

def prepare(data):
    X = data[:, 0:2]
    y = data[:,2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify = y)
    parameters = {'kernel': ['linear'], 'C':[0.1, 10]}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters, cv = 10)
    clf.fit(X_train, y_train)
    clf.cv_results_
    clf.score(X_train, y_train)
    
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    C = [0.1, 0.5, 1, 5, 10, 50, 100]
    print "C=", C
    print(" Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
def main(argv):
    input_file = 'C:/Users/alin/Documents/SelfStudy/MyLearning/AI/Project3/input3.csv'
#    input_file = 'input3.csv'
    data = np.loadtxt(input_file, delimiter = ',')
    prepare(data)



if __name__ == "__main__":
    main(sys.argv[1:])