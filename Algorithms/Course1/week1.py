#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:29:18 2017

@author: alin
"""
import numpy as np
def multiply(x, y, lx, ly):
   if lx < 0:
       lx = len(str(x))
   if ly < 0:
       ly = len(str(y))
   if lx == 1:
       if ly == 1:
           return x * y
       else:
           h_ly = int(ly/2)
           c = int(str(y)[:h_ly])
           d = int(str(y)[h_ly:])
           return multiply(x, c, lx, h_ly) * np.power(10, ly - h_ly) + multiply(x, d, lx, ly - h_ly)
   else:
       h_lx = int(lx/2)
       a = int(str(x)[:h_lx])
       b = int(str(x)[h_lx:])
       if ly == 1:
           return multiply(a, y, h_lx, ly) * np.power(10, lx - h_lx) + multiply(b, y, lx - h_lx, ly)
       else:
           h_ly = int(ly/2)
           c = int(str(y)[:h_ly])
           d = int(str(y)[h_ly:])
           ac = multiply(a, c, h_lx, h_ly)
           bd = multiply(b, d, lx - h_lx, ly - h_ly)
           ad = multiply(a, d, h_lx, ly - h_ly)
           bc = multiply(b, c, lx - h_lx, h_ly)
           return ac*np.power(10, lx - h_lx + ly - h_ly ) + bd + ad*np.power(10, lx - h_lx) + bc*np.power(10, ly - h_ly)
           
multiply(125323, 124322, -1, -1)           
