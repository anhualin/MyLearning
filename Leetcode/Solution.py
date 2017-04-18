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
        LS = len(s)
        if LS <= 1:
            return s
        curr = 0
        bestCenter = 0
        if s[0] == s[1]:
            bestLength = 2
        else:
            bestLength = 1
        curr += 1
        while curr + bestLength/2 < LS:
            #check odd palindrome centered at curr
            half = bestLength / 2 if bestLength % 2 == 0 else bestLength / 2 + 1
            if curr - half >= 0 and curr + half < LS and s[curr - half: curr] == s[curr + half: curr: -1]:
                half += 1
                while curr - half >=0 and curr + half < LS and s[curr - half] == s[curr + half]:
                    half += 1
                half -= 1
                bestLength = 2 * half + 1
                bestCenter = curr
            #check even palindrome centered at curr
            half = bestLength / 2 + 1
            if curr - half + 1 >= 0 and curr + half < LS and s[curr - half + 1: curr + 1] == s[curr + half: curr: -1]:
                half += 1
                while curr - half + 1 >= 0 and curr + half < LS and s[curr - half + 1] == s[curr + half]:
                    half +=1
                half -=1
                bestLength = 2 * half
                bestCenter = curr
            curr += 1
        half = bestLength / 2
        palindrome = s[bestCenter - half + 1: bestCenter + half + 1] if bestLength % 2 == 0 else s[bestCenter - half: bestCenter + half + 1]    
        return palindrome
                    
                    
sol = Solution()
print sol.longestPalindrome('cbbdxcdefggfedc12345689098654321c2')
