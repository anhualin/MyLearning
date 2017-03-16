#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:59:07 2017

@author: alin
"""

import numpy as np
from copy import copy
state0 = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'

def str2board(state):
    sudoku = {}
    for i in range(9):
        for j in range(9):
            sudoku[(i,j)] = int(state[i * 9 + j])
    return sudoku

def display(board):
    for i in range(9):
        print '\n'
        s = ''
        for j in range(9):
            s += str(board[i, j]) + ' '
        print s
sudoku = str2board(state0)
display(sudoku)

def setup(sudoku):
    unassigned = []
    assigned = []
    allowed = {}
    for i in range(9):
        for j in range(9):
            if sudoku[(i,j)] > 0:
                assigned.append((i,j))
                #allowed[(i,j)] = [sudoku[(i,j)]]
            else:
                unassigned.append((i,j))
                allowed[(i,j)] = range(1,10)
    for (i,j) in assigned:
       for k in range(9):
           if sudoku[(i,k)] == 0 and sudoku[(i,j)] in allowed[(i,k)]:
               allowed[(i,k)].remove(sudoku[(i,j)])
           if sudoku[(k,j)] == 0 and sudoku[(i,j)] in allowed[(k,j)]:
               allowed[(k,j)].remove(sudoku[(i,j)])
        
       x = (i/3)*3
       y = (j/3)*3
       for l in range(3):
           for m in range(3):
#               if x+l == 0 and y + m == 1:
#                   print (i,j), (l,m), (x,y)
               if sudoku[(x+l, y+m)] == 0 and sudoku[(i,j)] in allowed[(x+l, y+m)]:
                   allowed[(x+l, y+m)].remove(sudoku[(i,j)])
    return unassigned, assigned, allowed

unassigned, assigned, allowed = setup(sudoku)
a = [(k, len(allowed[k])) for k in allowed.keys()]
def choose_var(allowed):
    """ choose the unassigned with the fewest allowed values """
    key_len = [(k, len(allowed[k])) for k in allowed.keys()]
    
def backtrack(assigned, allowed, sudoku):
    if len(assigned) == 81:
        return sudoku
    v = choose_var(allowed)
    domain = order_domain(allowed, v)
    for val in domain:
        assigned.append(v)
        allowed1 = copy(allowed)
        sudoku[v] = val
        del allowed1[v]
        update(v, allowed1, sudoku1)
        result = backtract(assigned1, allowed1, sudoku1)
        if result:
            return result
        else:
            assigned.remove(v)
            sudoku[v] = 0
    return False
            