# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:56:17 2017

@author: alin
"""

import numpy as np
import sys

def perceptron(data, f):
    stop = False
    X = data[:,0:2]
    y = data[:, 2]
    N = data.shape[0]
    m = data.shape[1] - 1
    w = np.zeros(m)
    b = 0
    j = 0
    stop = False
    while not stop:
        j += 1
        stop = True
        for i in range(N):
            x = X[i]
            s = (b + sum(w * x)) * y[i]
            if s <= 0:
                w += y[i] * x
                b += y[i]
                stop = False
            f.write(str(w[0]) + ',' + str(w[1]) + ',' + str(b) + '\n')
    f.close()


def main(argv):
    input_file = argv[0]
    data = np.loadtxt(input_file, delimiter = ',')
    output = argv[1]
    f = open(output, 'w')

    perceptron(data, f)

if __name__ == "__main__":
    main(sys.argv[1:])