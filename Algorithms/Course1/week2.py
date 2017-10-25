#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:07:33 2017

@author: alin
"""

def Sort_Inv(x):
    """
    input: x in a list of non-repetive numbers.
    output: 1. number of inversions, i.e. x[i] > x[j]
            2. a sorted version of x
    """
    n = len(x)
    if n == 1:
        return x, 0
    x_l, inv_l = Sort_Inv(x[:int(n/2)])
    x_r, inv_r = Sort_Inv(x[int(n/2):])
    x_sort, inv = Merge_split_inv(x_l, x_r)
    return x_sort, inv_l + inv_r + inv

def Merge_split_inv(x, y):
    """ 
    input: sorted list x and y
    output: 1. sorted list of all elements in x and y
            2. numnber of pairs (a, b) where a in x, b in y and  a > b
    """
    out = []
    lx = len(x)
    ly = len(y)
    inv = 0
    i = 0
    j = 0
    for k in range(lx + ly):
        if i == lx:
            out += y[j:]
            return out, inv
        if j == ly:
            out += x[i:]
            return out, inv
        if x[i] < y[j]:
            out.append(x[i])
            i += 1
        else:
            out.append(y[j])
            inv += lx - i
            j += 1
    return out, inv

def brute_force_inv(x):
    inv = 0
    lx = len(x)
    for i in range(lx - 1):
        c = x[i]
        for y in x[i+1:]:
            if c > y:
                inv += 1
    return inv

x = [3, 1, 2, 5, 8]
a, b = Sort_Inv(x)
c = brute_force_inv(x)

import numpy as np
import time
N = 20000
x = [i for i in range(N)]
x = list(np.random.choice(x, size=N, replace=False))
t1 =time.time()
inv_0 = brute_force_inv(x)
t2 = time.time()
y, inv_1 = Sort_Inv(x)
t3 = time.time()
print("diff=", inv_0 - inv_1)
print("time1=", t2 - t1)
print('time2=', t3 - t2)
