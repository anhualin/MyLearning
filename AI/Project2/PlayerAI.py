#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:21:28 2017

@author: alin
"""

from Displayer  import Displayer
from Grid       import Grid
from BaseAI import BaseAI


class PlayerAI(BaseAI):
#    def getMove(self, grid):
#        moves = grid.getAvailableMoves()
#        return moves[randint(0, len(moves) - 1)] if moves else None    
    def __init__(self):
        self.possibleNewTiles = [2, 4]
       
    def orderMoves(self, grid, moves):
        moves1 = [(grid.clone(), moves[i]) for i in range(len(moves))]
        for item in moves1:
            item[0].move(item[1])
        moves2 = [(e[0].getMaxTile(), e[1]) for e in moves1]
        moves3 = sorted(moves2, key = lambda x: -x[0])
        moves4 = [e[1] for e in moves3]
        return moves4
        
                
                  
    def getMove(self):
        grid = Grid(2)
        grid.setCellValue((0,0),2)
        grid.setCellValue((0,1),4)
#        grid.setCellValue((1,0),2)
        grid.setCellValue((1,1),4)
        displayer 	= Displayer()
        moves = grid.getAvailableMoves()
        bestMove = None
        level = {'player': 'p', 'val': 0, 'grid': grid, 'moves': moves, 'mv_ind': 0}
        stack = [level]
        maxdep = 0
        while stack:
            if maxdep < len(stack):
                maxdep = len(stack)
            current = stack[0]
           
            displayer.display(current['grid'])
            print current['player']
            print current['val']
            s = raw_input('--->')
            if current['mv_ind'] == len(current['moves']):
                children = stack.pop(0)
                print 'pop\n'
                if stack:
                    father = stack[0]
                    if father['player'] == 'p':
                        if children['val'] > father['val']:
                            father['val'] = children['val']
                            if len(stack) == 1:
                                bestMove = father['moves'][father['mv_ind']] 
                    else:
                        if children['val'] < father['val']:
                            father['val'] = children['val']
                    father['mv_ind'] += 1
            else:
                pruned = False
                if len(stack) > 1:
                    # try pruning
                    father_val = stack[1]['val']
                    if (current['player'] == 'c' and current['val'] <= father_val) or (current['player'] == 'p' and current['val'] >= father_val):
                        current['mv_ind'] = len(current['moves'])
                        pruned = True
                if not pruned:
                    if current['player'] == 'p': 
                        #computer's turn
                        gridCopy = current['grid'].clone()
                        gridCopy.move(current['moves'][current['mv_ind']])
                        cells = gridCopy.getAvailableCells()
                        print cells
                        # cells cannot be empty                        
                        moves = [(2, c) for c in cells ] + [(4, c) for c in cells]
                        level = {'player': 'c', 'val': 16*20480, 'grid':gridCopy, 'moves': moves, 'mv_ind': 0}
                        stack.insert(0, level)
                    else:
                        #player's turn
                        gridCopy = current['grid'].clone()
                        cellSet = current['moves'][current['mv_ind']]
                        cellValue = cellSet[0]
                        print '888\n'
                        print cellSet
                        print cellValue
                        print '***\n'
                        cellLoc = cellSet[1]
                        gridCopy.setCellValue(cellLoc, cellValue)
                        moves = gridCopy.getAvailableMoves()
                        if not moves:
                            # cannot move any more, leaf node
                            maxTile = gridCopy.getMaxTile()
                            if maxTile < current['val']:
                                current['val'] = maxTile
                            current['mv_ind'] += 1
                        else:
                            level = {'player': 'p', 'val': 0, 'grid': gridCopy, 'moves': moves,
                                     'mv_ind': 0}
                            stack.insert(0, level)
        print children
        print maxdep
        return bestMove
        
def main():
    p = PlayerAI()
    p.getMove()
    

if __name__ == '__main__':
    main()

