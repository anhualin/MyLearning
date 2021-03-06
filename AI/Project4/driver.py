#!/usr/bin/env python2x
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 16:59:07 2017

@author: alin
"""

import sys
from copy import deepcopy

def str2board(state):
    sudoku = {}
    for i in range(9):
        for j in range(9):
            sudoku[(i,j)] = int(state[i * 9 + j])
    return sudoku
def board2str(sudoku):
    state = ''
    for i in range(9):
        for j in range(9):
            state += str(sudoku[(i,j)])
    return state
    
def display(board):
    for i in range(9):
        print '\n'
        s = ''
        for j in range(9):
            s += str(board[i, j]) + ' '
        print s

def getNeighbor():
    neighbor = {}
    for i in range(9):
        for j in range(9):
            neighbor[(i,j)] = set([])
            for k in range(9):
                if k != j:
                    neighbor[(i,j)].add((i,k))
                if k != i:
                    neighbor[(i,j)].add((k,j))
            x = (i/3)*3
            y = (j/3)*3
            for l in range(3):
                for m in range(3):
                    if x+l != i and y+m != j:
                        neighbor[(i,j)].add((x+l,y+m))
    return neighbor

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
    neighbor = getNeighbor()
    for (i,j) in assigned:
        for n in neighbor[(i,j)]:
            if sudoku[n] == 0 and sudoku[(i,j)] in allowed[n]:
                allowed[n].remove(sudoku[(i,j)])
    return unassigned, assigned, allowed, neighbor


def choose_var(allowed):
    """ choose the unassigned with the fewest allowed values """
    key_len = [(k, len(allowed[k])) for k in allowed.keys()]
    min_len = min([l for (k, l) in key_len])
    k_selected = [k for (k,l) in key_len if l == min_len][0]
    return k_selected                      

def order_domain(allowed, v, neighbor, sudoku):
    g = []
    for val in allowed[v]:
        num_affected = len([x for x in neighbor[v] if sudoku[x] == 0 and val in allowed[x]])
        g.append((val, num_affected))
    h = sorted(g, key = lambda (x,y) : y)
    return [x for (x,y) in h]                 

    
def backtrack(assigned, allowed, sudoku, neighbor):
   
    if len(assigned) == 81:
        return sudoku
    v = choose_var(allowed)
  
    
    domain = order_domain(allowed, v, neighbor, sudoku)
   
   
    assigned.append(v)
    for val in domain:
      
        allowed1 = deepcopy(allowed)
        sudoku[v] = val
        del allowed1[v]
        for n in neighbor[v]:
            if sudoku[n] == 0 and val in allowed1[n]:
                allowed1[n].remove(val)
        result = backtrack(assigned, allowed1, sudoku, neighbor)
        if result:
            return result
    assigned.remove(v)
    sudoku[v] = 0
  
    return False



#def load_data():
#    #start_file = 'C:/Users/alin/Documents/SelfStudy/MyLearning/AI/Project4/sudokus_start.txt'
#    start_file = 'sudokus_start.txt'
#    with open(start_file) as f:
#        sudoku_starts = f.readlines()
#    sudoku_starts = [x.strip() for x in sudoku_starts]
#                     
#    #finish_file = 'C:/Users/alin/Documents/SelfStudy/MyLearning/AI/Project4/sudokus_finish.txt'
#    finish_file = 'sudokus_finish.txt'
#    with open(finish_file) as f:
#        sudoku_finishes = f.readlines()
#    sudoku_finishes = [x.strip() for x in sudoku_finishes]
#    return sudoku_starts, sudoku_finishes
#sudoku_starts, sudoku_finishes = load_data()                 
#
#state0 = sudoku_starts[3]
#sudoku = str2board(state0)
#display(sudoku)
#unassigned, assigned, allowed, neighbor = setup(sudoku)
#sboard = backtrack(assigned, allowed, sudoku, neighbor)

def main(argv):
    state = argv[0]
    sudoku = str2board(state)
    unassigned, assigned, allowed, neighbor = setup(sudoku)
    solution = backtrack(assigned, allowed, sudoku, neighbor)
    sol_str = board2str(solution)
    f = open('output.txt', 'w')
    f.write(sol_str)
    f.close()
if __name__ == "__main__":
    main(sys.argv[1:])
    
    
#total_error = 0
#for i in range(len(sudoku_starts)):
#    sudoku = str2board(sudoku_starts[i])
#    unassigned, assigned, allowed, neighbor = setup(sudoku)
#    solution = backtrack(assigned, allowed, sudoku, neighbor)
#    if solution:
#        sol_str = board2str(solution)
#        if sol_str != sudoku_finishes[i]:
#            total_error += 1
#    else:
#        total_error += 1
#print "total_error = ", total_error
