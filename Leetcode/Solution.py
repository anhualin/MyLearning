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
        if len(s) == 0:
            return 0
        i = 0
        for i in range(len(s)):
            if s[i] != ' ':
                break
            print i
       
        print i
        return i
sol = Solution()
print sol.myAtoi('\t')
