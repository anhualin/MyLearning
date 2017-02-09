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
       
        
    def getMove(self):
        grid = Grid(2)
        grid.setCellValue((0,0),2)
        displayer 	= Displayer()
        moves = grid.getAvailableMoves()
        bestMove = None
        level = {'player': 'p', 'val': 0, 'grid': grid, 'moves': moves, 'mv_ind': 0}
        stack = [level]
        while stack:
        
            current = stack[0]
           
            displayer.display(current['grid'])
            print current['player']
            s = raw_input('--->')
            if current['mv_ind'] == len(current['moves']):
                children = stack.pop(0)
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
                if current['player'] == 'p': 
                    #computer's turn
                    gridCopy = current['grid'].clone()
                    gridCopy.move(current['moves'][current['mv_ind']])
                    cells = gridCopy.getAvailableCells()
                    # cells cannot be empty                        
                    moves = [(2, c) for c in cells ] + [(4, c) for c in cells]
                    level = {'player': 'c', 'val': 16*20480, 'grid':gridCopy, 'moves': moves, 'mv_ind': 0}
                    stack.insert(0, level)
                else:
                    #player's turn
                    gridCopy = current['grid'].clone()
                    cellSet = current['moves'][current['mv_ind']]
                    cellValue = cellSet[0]
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
        return bestMove
        
def main():
    p = PlayerAI()
    p.getMove()
    

if __name__ == '__main__':
    main()

