#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:00:52 2017

@author: alin
"""

class Solution(object):
  def myAtoi(self, s):
        """
        :type str: str
        :rtype: int
        """
        digits = [str(d) for d in range(10)]
        if len(s) == 0:
            return 0
        i = 0
        
        for i in range(len(s)):
            if s[i] != ' ':
                break
           
        
        if i < len(s):
            pm = 1
            if s[i] == '+':
                i += 1
            elif s[i] == '-':
                pm = -1
                i += 1
            if i == len(s) or s[i] not in  digits:
                return 0
            a = 0
            while(i < len(s) and s[i] in digits):
                a = 10 * a + int(s[i])
                i += 1
            a = pm * a
            a = max(min(a, 2147483647), -2147483648)
            return a
        else:
            return 0
sol = Solution()
print sol.myAtoi('')
