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
moveTimeLimit = 0.08


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
            moves2 = [(e[0].getMaxTile(), e[1]) for e in moves1]
            moves3 = sorted(moves2, key = lambda x: -x[0])
            moves4 = [e[1] for e in moves3]
            return moves4, moves3[0][0]
        else:
            return [], 0

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
        moves1 = [(grid.clone(), move) for move in moves]
        for item in moves1:
            item[0].setCellValue(item[1][1], item[1][0])

        moves2 = [(self.orderMoves(e[0])[1], e[1]) for e in moves1]
        moves3 = sorted(moves2, key = lambda x: x[0])
        moves4 = [e[1] for e in moves3]
        return moves4

    def getMove(self, grid):
        """ Input: a grid
            Output: the estimated best move
        """
        startTime = time.clock()
        initMoves = self.orderMoves(grid)[0]
        bestMove = None
        bestVal = 0
        depthBound = 2
        while(True):
            print 'depthBound = ', depthBound
            #each round we start from beginning to the given depth bound
            level = {'player': 'p', 'val': 0, 'grid': grid.clone(), 'moves': initMoves, 'mv_ind': 0}
            stack = [level]
            currentBestMove = None
            currentBestVal = 0
            maxDepth = 1
            iter = 0
            while stack:
                iter +=1
                print 'iter = ', iter
                if time.clock() - startTime > moveTimeLimit:
                    #time's up
                    print 'damn'
                    return bestMove
                current = stack[0]
                if current['mv_ind'] == len(current['moves']):
                    # given the current grid, all possible chocies have been explored
                    children = stack.pop(0)
                    if stack:
                        father = stack[0]
                        if father['player'] == 'p':
                            if children['val'] > father['val']:
                                # this is the currently best move for father
                                father['val'] = children['val']
                                if len(stack) == 1:
                                    # at the first level, update the current choice
                                    currentBestMove = father['moves'][father['mv_ind']]
                                    currentBestVal = father['val']
                        else:
                            if children['val'] < father['val']:
                                # this is the current best cell choise for father (computer)
                                father['val'] = children['val']
                                if father['val'] <= father['grid'].getMaxTile():
                                    # for type c, the max tile of grid is the lowerbound
                                    father['mv_ind'] = len(father['moves'])-1
                        father['mv_ind'] += 1
                else:
                    pruned = False
                    if len(stack) > 1:
                    # try pruning
                        father_val = stack[1]['val']
                        if (current['player'] == 'c' and current['val'] <= father_val) or (current['player'] == 'p' and current['val'] >= father_val):
                        #current['mv_ind'] = len(current['moves'])
                            stack.pop(0)
                            stack[0]['mv_ind'] += 1
                            pruned = True
                    if not pruned:
                        if current['player'] == 'p':
                            #player's turn
                            gridCopy = current['grid'].clone()
                            gridCopy.move(current['moves'][current['mv_ind']])
                            #add the next layer cell choices
                            moves = self.orderCells(gridCopy)
                            level = {'player': 'c', 'val': 16*20480, 'grid':gridCopy, 'moves': moves, 'mv_ind': 0}
                            stack.insert(0, level)
                        else:
                            #computer's turn
                            gridCopy = current['grid'].clone()
                            cellSet = current['moves'][current['mv_ind']]
                            cellValue = cellSet[0]
                            cellLoc = cellSet[1]
                            gridCopy.setCellValue(cellLoc, cellValue)
                            #add the next layer player's choices
                            moves, maxVal = self.orderMoves(gridCopy)
                            if not moves:
                                # cannot move any more, leaf node
                                maxTile = gridCopy.getMaxTile()
                                if maxTile < current['val']:
                                    current['val'] = maxTile
                                if maxTile == current['grid'].getMaxTile():
                                    # this is the case when
                                    # computer can add a number and
                                    # the game is over
                                    current['mv_ind'] = len(current['moves'])
                                else:
                                    current['mv_ind'] += 1
                            elif len(stack) >= depthBound:
                                #reach current bound
                                #use heuristic to estimate the leaf value
                                current['val'] = self.estimateMaxVal(gridCopy)  #current naive heuristic
                                current['mv_ind'] += 1
                            else:
                                #add players moves
                                level = {'player': 'p', 'val': 0, 'grid': gridCopy, 'moves': moves,
                                         'mv_ind': 0}
                                stack.insert(0, level)
                                maxDepth = len(stack)
            if currentBestVal > bestVal:
                bestVal = currentBestVal
                bestMove = currentBestMove
            print 'maxDepth = ', maxDepth
            if maxDepth < depthBound:
                #have searched the wohle true
                return bestMove
            if time.clock() - startTime > moveTimeLimit:
                return bestMove
            depthBound += 100
        return bestMove

    def estimateMaxVal(self, grid):
        #heuristic function to estimate the max final value for the given grid
        _, maxVal = self.orderMoves(grid)
        return maxVal
##def main():
##    p = PlayerAI()
##    move, val = p.getMove()
##    print 'best move is ', move
##    print 'best value is ', val
###    p.test()
##
##if __name__ == '__main__':
##    main()

