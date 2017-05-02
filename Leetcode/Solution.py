#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:00:52 2017

@author: alin
"""

class Solution(object):
   def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows <= 1:
            return s
        zig = ['' for i in range(len(s))]
        current = 0
        for i in range(numRows):
            if i == 0:
                step = [2*numRows - 2, 2*numRows - 2]
            elif i == numRows - 1:
                step = [2*i, 2*i]
            else:
                step = [2*(numRows - 1 - i), 2*i]
            
            assign = i
            j = 0
            while assign < len(s):
                zig[current] = s[assign]
                assign += step[j]
                current += 1
                j = (j+1) % 2
         
        return ''.join(zig)
                    
sol = Solution()
print sol.convert("PAYPALISHIRING", 2)
