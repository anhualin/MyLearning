#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:00:52 2017

@author: alin
"""

class Solution(object):
     def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) <= 1:
            return s
        curr = 0
        bestCenter = 0
        if s[0] == s[1]:
            bestLength = 2
        else:
            bestLength = 1
        curr += 1
        while curr + bestLength/2 < len(s):
            if bestLength % 2 == 0:
                if s[curr - bestLength /2: curr ] == s[curr+1: curr + bestLength/2]:
                    
            