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
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#import sys

input_file = 'input3.csv'
data = np.loadtxt(input_file, delimiter = ',')

f = open('output3.csv', 'w')

X = data[:, 0:2]
y = data[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, stratify = y)

#1
#method = 'svm_linear'
#parameters = {'kernel': ['linear'], 'C':[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}

#2
method = 'svm_polynomial'    
parameters = {'kernel': ['poly'], 'C':[0.1, 1.0, 3.0],
              'degree': [4, 5, 6], 'gamma': [0.1, 1.0]}


method = 'svm_rbf'
parameters = {'kernel': ['rbf'], 'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
              'gamma': [0.1, 0.5, 1.0, 3.0, 6.0, 10.0]}


method = 'logistic'
parameters = {'C': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}

method = 'knn'
parameters = {'n_neighbors':range(1,51),
              'leaf_size': range(5,65,5)}

method = 'decision_tree'
parameters = {'max_depth': range(1,51), 
              'min_samples_split': range(2,11)}

method = 'random_forest'
parameters = {'max_depth': range(1,51), 
              'min_samples_split': range(2,11)}

#svr = svm.SVC()
#logit = LogisticRegression()
#knn = KNeighborsClassifier()

#dtree = DecisionTreeClassifier()

rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, cv = 5)
clf.fit(X_train, y_train)
best_score = clf.best_score_
best_estimator = clf.best_estimator_
test_score = clf.score(X_test, y_test)
f.write(method + ',' + str(best_score) + ',' + str(test_score) + '\n')

# svm_polynomial, 0.71666666666666667,  0.69499999999999995
# svm_rbf, 0.94999999999999996, 0.95499999999999996
# logistic, 0.56333333333333335, 0.58499999999999996
# knn,  0.91333333333333333, 0.94499999999999995
# decision_tree, 0.97666666666666668, 1.0


f.close()
def main(argv):
    #input_file = 'C:/Users/alin/Documents/SelfStudy/MyLearning/AI/Project3/input3.csv'
    
    prepare(data)



if __name__ == "__main__":
    main(sys.argv[1:])