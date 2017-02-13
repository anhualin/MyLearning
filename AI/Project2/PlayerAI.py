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
moveTimeLimit = 0.1


class PlayerAI(BaseAI):
#    def getMove(self, grid):
#        moves = grid.getAvailableMoves()
#        return moves[randint(0, len(moves) - 1)] if moves else None    
    def __init__(self):
        self.possibleNewTiles = [2, 4]
       
    def orderMoves(self, grid):
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
        cells = grid.getAvailableCells()
        moves = [(2, c) for c in cells ] + [(4, c) for c in cells]
        moves1 = [(grid.clone(), move) for move in moves]
        for item in moves1:
            item[0].setCellValue(item[1][1], item[1][0])
            
        moves2 = [(self.orderMoves(e[0])[1], e[1]) for e in moves1]
        moves3 = sorted(moves2, key = lambda x: x[0])
        moves4 = [e[1] for e in moves3]
        return moves4
        
    def test(self):
        grid = Grid(2)
        grid.setCellValue((0,0),4)
        grid.setCellValue((0,1),6)
#        grid.setCellValue((1,0),2)
#        grid.setCellValue((1,1),8)
#        moves = grid.getAvailableMoves()
#        cells = grid.getAvailableCells()
#        moves = [(2, c) for c in cells ] + [(4, c) for c in cells]
        moves  = self.orderCells(grid)
        print moves
        
    def getMove(self):
        prevTime = time.clock()
        grid = Grid(2)
        grid.setCellValue((0,0),2)
        grid.setCellValue((0,1),16)
        grid.setCellValue((1,0),16)
        grid.setCellValue((1,1),0)
##        
        
#        grid.setCellValue((0,0),16)
#        grid.setCellValue((0,1),2)
#        grid.setCellValue((1,0),4)
#        grid.setCellValue((1,1),2)
        
        displayer 	= Displayer()
        initMoves = self.orderMoves(grid)[0] 
        bestMove = None
        bestVal = 0
        #level = {'player': 'p', 'val': 0, 'grid': grid, 'moves': moves, 'mv_ind': 0}
        maxAllowedDepth = 20
        depthBound = 1
        while(True):
            level = {'player': 'p', 'val': 0, 'grid': grid.clone(), 'moves': initMoves, 'mv_ind': 0}          
            stack = [level]    
            currentBestMove = None
            currentBestVal = 0
            print "before start depthBound = ", depthBound
            print stack[0]
            maxDepth = 1
            while stack: 
                if time.clock() - prevTime > moveTimeLimit:
                    return bestMove, bestVal
                print "inner loop ", len(stack)
                print "depthBound =", depthBound
                current = stack[0]
                print current['moves']
                print current['mv_ind']
#            displayer.display(current['grid'])
#            print current['player']
#            print current['val']
#            print 'dep = ', len(stack)
#            s = raw_input('--->')
                if current['mv_ind'] == len(current['moves']):
                    children = stack.pop(0)
                    print 'pop\n'
              
                    if stack:
                        father = stack[0]
                        if father['player'] == 'p':
                            if children['val'] > father['val']:
                                father['val'] = children['val']
                                if len(stack) == 1:
                                    currentBestMove = father['moves'][father['mv_ind']]
                                    currentBestVal = father['val']
                        else:
                            if children['val'] < father['val']:
                                father['val'] = children['val']
                                if father['val'] <= father['grid'].getMaxTile():
                                    # for type c, the max tile of grid is the lowerbound
                                    father['mv_ind'] = len(father['moves'])-1
                        father['mv_ind'] += 1
                    print "after pop\n"
                    print len(stack)
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
                            print "pruned"
                    if not pruned:
                        if current['player'] == 'p': 
                            #player's turn
                            gridCopy = current['grid'].clone()
                            gridCopy.move(current['moves'][current['mv_ind']])
                            moves = self.orderCells(gridCopy)
                            print moves
    #                        # cells cannot be empty                        
    #                        moves = [(2, c) for c in cells ] + [(4, c) for c in cells]
                            #add computer moves
                            level = {'player': 'c', 'val': 16*20480, 'grid':gridCopy, 'moves': moves, 'mv_ind': 0}
                            stack.insert(0, level)
                        else:
                            #computer's turn
                            gridCopy = current['grid'].clone()
                            cellSet = current['moves'][current['mv_ind']]
                            cellValue = cellSet[0]
                            print '888\n'
                            print cellSet
                            print cellValue
                            print '***\n'
                            cellLoc = cellSet[1]
                            gridCopy.setCellValue(cellLoc, cellValue)
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
            print "just got out"
            if currentBestVal > bestVal:
                bestVal = currentBestVal
                bestMove = currentBestMove 
            if maxDepth < depthBound:
                #have searched the wohle true
                return bestMove, bestVal
            if time.clock() - prevTime > moveTimeLimit:
                return bestMove, bestVal
            depthBound += 4                   
        return bestMove, bestVal

    def estimateMaxVal(self, grid):
        #heuristic function to estimate the max final value for the given grid
        _, maxVal = self.orderMoves(grid)
        return maxVal
def main():
    p = PlayerAI()
    move, val = p.getMove()
    print 'best move is ', move
    print 'best value is ', val
#    p.test()

if __name__ == '__main__':
    main()

