#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 22:21:28 2017

@author: alin
"""

from random import randint
from BaseAI import BaseAI
from copy import deepcopy

class PlayerAI(BaseAI):
    def getMove(self, grid):
        
        moves = grid.getAvailableMoves()
        if not moves:
            return None
        maxVal = 0
        for move in moves:
            _, value = self.ComputerMove(deepcopy(grid).move(move))
            if value > maxVal:
                bestMove = move
                maxVal = value
        return bestMove
    
    def ComputerMove(self,)