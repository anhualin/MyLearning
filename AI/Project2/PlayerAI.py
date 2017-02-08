#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:21:28 2017

@author: alin
"""

from random import randint
from BaseAI import BaseAI


class PlayerAI(BaseAI):
#    def getMove(self, grid):
#        moves = grid.getAvailableMoves()
#        return moves[randint(0, len(moves) - 1)] if moves else None    
    def __init__(self):
        self.possibleNewTiles = [2, 4]
    def getMove(self, grid):
        move, value = self.maximize(grid)
        return move
    def maximize(self, grid):
        moves = grid.getAvailableMoves()
        maxVal = 0
        bestMove = None
        level = {'player': 'p', 'val': maxVal, 'grid': grid, 'moves': moves, 'mv_ind': 0}
        stack = [level]
        while stack:
            current = stack[0]
            if current['mv_ind'] == len(current['moves']):
                children = stack.pop(0)
                if stack:
                    father = stack[0]
                    if father['player'] == 'p':
                        if children['val'] > father['val']:
                            father['val'] = children['val']
                    else:
                        if children['val'] < father['val']:
                            father['val'] = children['val']
                else:
                    #stack empty
                    
                father = stack[0]
                bestMove = current['move']
                bestVal = current['val']
            else:
                if current['player'] == 'p': 
                    #computer's turn
                    gridCopy = grid.clone()
                    gridCopy.move(current['moves'][current['mv_ind']])
                    cells = gridCopy.getAvailableCells()
                    # cells cannot be empty                        
                    moves = [(2, c) for c in cells ] + [(4, c) for c in cells]
                    level = {'player': 'c', 'val': 16*20480, 'grid':gridCopy, 'moves': moves, 'mv_ind': 0}
                    stack.insert(0, level)
                else:
                    #player's turn
                    moves = grid.getAvailableMoves()
                    if not moves:
                        # cannot move any more, leaf node
                        maxTile = grid.getMaxTile()
                        if maxTile < current['val']:
                            current['val'] = maxTile
                        current['mv_ind'] += 1
                    else:
                        gridCopy = grid.clone()
                        move = current['moves'][current['mv_ind']]
                        cellValue = move[0]
                        cellLoc = move[1]
                        gridCopy.setCellValue(cellLoc, cellValue)
                        level = {'player': 'p', 'val': 0, 'grid': gridCopy, 'moves': moves,
                                 'mv_ind': 0}
                        stack.insert(0, level)
       