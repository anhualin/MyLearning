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
        for mv in moves:
            gridCopy = grid.clone()
            gridCopy.move(mv)
            value = self.minimize(gridCopy)
            if value > maxVal:
                bestMove = mv
                maxVal = value
        return bestMove, maxVal
    
    def minimize(self, grid):
        cells = grid.getAvailableCells()
        minValue = 16*20480
        for tile in self.possibleNewTiles:
            for cell in cells:
                gridCopy = grid.clone()
                gridCopy.setCellValue(cell, tile)
                _, value = self.maximize(gridCopy)
                if value < minValue:
                    minValue = value
                
        return minValue