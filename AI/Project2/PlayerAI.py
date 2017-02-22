#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:21:28 2017

@author: alin
"""

from Displayer  import Displayer
from Grid       import Grid
from BaseAI import BaseAI
import time
from math import log
moveTimeLimit = 0.09


class PlayerAI(BaseAI):

    def __init__(self):
        self.possibleNewTiles = [2, 4]

    def orderMoves(self, grid):
        """ Input: a grid
            Output: (i) the list of all possible moves ordered by descreasing maxTile
                if the grid perorms the correspondign move
                    (ii) the biggest maxTile of after these moves
        """
        moves0 = grid.getAvailableMoves()
        if moves0:
            moves1 = [(grid.clone(), move) for move in moves0]
            for item in moves1:
                item[0].move(item[1])
            moves2 = [(e[0].getMaxTile(), e[0], e[1]) for e in moves1]
            moves3 = sorted(moves2, key = lambda x: -x[0])
            moves4 = [e[1] for e in moves3]
            return moves4, moves3[0][0]
        else:
            return [], grid.getMaxTile()

    def initOrderMoves(self, grid):
        """ Input: a grid
            Output: (i) the list of all possible moves ordered by descreasing maxTile
                if the grid perorms the correspondign move
                    (ii) the biggest maxTile of after these moves
        """
        moves0 = grid.getAvailableMoves()
        if moves0:
            moves1 = [(grid.clone(), move) for move in moves0]
            for item in moves1:
                item[0].move(item[1])
            moves2 = [(e[0].getMaxTile(), e[0], e[1]) for e in moves1]
            moves3 = sorted(moves2, key = lambda x: -x[0])
            moves4 = [(e[1], e[2]) for e in moves3]
            return moves4
        else:
            return [], grid.getMaxTile()

    def orderCells(self, grid):
        """ Input: a grid
            Output: the list of all possible (val, blank cell)s.
                    for each (val, cell),
                    (i) set the cell val to be val.
                    (ii) find the max tile among all possible player's moves
                    (iii) order (val, cell) by this max tile value in increasing order
        """
        cells = grid.getAvailableCells()
        moves = [(2, c) for c in cells ] + [(4, c) for c in cells]
        score = []
        for v, pos in moves:
            grid.setCellValue(pos, v)
            score.append(-self.smoothness(grid))
            grid.setCellValue(pos, 0)
        maxScore = max(score)
        rest = [e for e in score if e != maxScore]
        if rest:
            secondScore = max(rest)
        else:
            secondScore = 0
        selectedMoves = []
        for i in range(len(moves)):
            if score[i] >= secondScore:
                selectedMoves.append(moves[i])
        return selectedMoves

    def getMove(self, grid):
        """ Input: a grid
            Output: the estimated best move
        """
        startTime = time.clock()
        alpha = -100
        beta = 1000000
        level = 1
        children = self.initOrderMoves(grid)
        #displayer = Displayer()
        bestVal = 0
        bestMove = None
        depthBound = 5
        while(True):
            level = 1
            for child, move in children:
                val = self.minimize(child, alpha, beta, startTime, level + 1, depthBound)
                if val > bestVal:
                    bestVal = val
                    bestMove = move
                if time.clock() - startTime > moveTimeLimit:
                    return bestMove
                alpha = max(alpha, bestVal)
            depthBound += 5
        return bestMove


    def maximize(self, grid, alpha, beta, startTime, level, depthBound):
        if level >= depthBound:
            return self.estimateVal(grid)
        if time.clock() - startTime > moveTimeLimit:
            return self.estimateVal(grid)
        children, _ = self.orderMoves(grid)
        if not children:
            #terminal
            return self.estimateVal(grid)
        maxVal = 0
        for child in children:
            val = self.minimize(child, alpha, beta, startTime, level + 1, depthBound)
            maxVal = max(maxVal, val)
            alpha = max(alpha, maxVal)
            if alpha >= beta:
                break

        return maxVal

    def minimize(self, grid, alpha, beta, startTime, level, depthBound):
        if level >= depthBound:
            return self.estimateVal(grid)
        if time.clock() - startTime > moveTimeLimit:
            return self.estimateVal(grid)
        children = self.orderCells(grid)
        minVal = 10000000
        for child in children:
            v = child[0]
            pos = child[1]
            grid.setCellValue(pos, v)
            val = self.maximize(grid, alpha, beta, startTime, level + 1, depthBound)
            grid.setCellValue(pos, 0)
##            print "val in min = ", val
            minVal = min(minVal, val)
            beta = min(beta, minVal)
            if beta <= alpha:
               # print "beta prune"
                break

        return minVal



    def estimateVal(self, grid):
        #heuristic function to estimate the max final value for the given grid
        return self.smoothness(grid)*0.6 + self.monotonicity(grid)*1.0 + log(len(grid.getAvailableCells())+1, 2) * 2.7+ grid.getMaxTile() * 1.0
#        return self.smoothness(grid)*0.1 + self.monotonicity(grid)*1.0 + log(len(grid.getAvailableCells())+1, 2) * 2.7+ grid.getMaxTile() * 1.0
    def monotonicity(self, grid):
        mvalue = [0, 0, 0, 0]
        for i in range(4):
            j = 0
            while j <= 2 and grid.getCellValue((i,j)) == 0:
                j += 1

            if j <= 2:
                s = log(grid.getCellValue((i,j)), 2)
                while j <= 2:
                    v = grid.getCellValue((i, j + 1))
                    d = 0 if v == 0 else log(v, 2)
                    if s > d:
                        mvalue[0]  = mvalue[0] + (s - d)
                    else:
                        mvalue[1] = mvalue[1] + (d - s)
                    j += 1
                    s = d

            j = 0
            while j <= 2 and grid.getCellValue((j,i)) == 0:
                j += 1

            if j <= 2:
                s = log(grid.getCellValue((j, i)), 2)
                while j <= 2:
                    v = grid.getCellValue((j + 1, i))
                    d = 0 if v == 0 else log(v, 2)
                    if s > d:
                        mvalue[2]  = mvalue[2] + (s - d)
                    else:
                        mvalue[3] = mvalue[3] + (d - s)
                    j += 1
                    s = d

        return max(mvalue[0], mvalue[1]) + max(mvalue[2], mvalue[3])
    def smoothness(self, grid):
        score = 0
        for i in range(4):
            j = 0
            while j <= 2 and grid.getCellValue((i,j)) == 0:
                j += 1
            if j <= 2:
                s = log(grid.getCellValue((i,j)), 2)
                while j <= 2:
                    j += 1
                    while j <= 3 and grid.getCellValue((i,j)) == 0:
                        j += 1
                    if j <= 3:
                        d = log(grid.getCellValue((i,j)),2)
                        score = score - abs(d - s)
                        s = d

            j = 0
            while j <= 2 and grid.getCellValue((j,i)) == 0:
                j += 1
            if j <= 2:
                s = log(grid.getCellValue((j,i)), 2)
                while j <= 2:
                    j += 1
                    while j <= 3 and grid.getCellValue((j,i)) == 0:
                        j += 1
                    if j <= 3:
                        d = log(grid.getCellValue((j,i)),2)
                        score = score - abs(d - s)
                        s = d
        return score
##    def isgroups(self, grid):
####        cnt = 0
####        gridCopy = grid.clone()
####        for i in range(4):
####            for j in range(4):
####                c = gridCopy.getCellValue((i,j))
####                if c != 0:
####                    cnt += 1
####                    if j < 3 and c == gridCopy.getCellValue((i, j+1)):
####                        gridCopy.setCellValue((i,j+1), 0)
####                    if i < 3 and c == gridCopy.getCellValue((i+1,j)):
####                        gridCopy.setCellValue((i+1, j), 0)
##        grp = set()
##        for i in range(4):
##            for j in range(4):
##                grp.add(grid.getCellValue((i,j)))
##        return len(grp)
##
##    def test(self, grid):
##       print self.isgroups(grid)

##def main():
##    p = PlayerAI()
##    grid = Grid(4)
##    grid.setCellValue((0,0), 2)
##    grid.setCellValue((1,0), 2)
##    grid.setCellValue((1,1), 4)
##    grid.setCellValue((1,3), 8)
##    p.test(grid)
####    move = p.getMove(grid)
####    print 'best move is ', move
##
###    p.test()
##
##if __name__ == '__main__':
##    main()

