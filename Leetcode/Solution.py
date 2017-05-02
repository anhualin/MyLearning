#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:00:52 2017

@author: alin
"""

class Solution(object):
  def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = -1 if x < 0  else 1
        a = str(x)
        y = a[1:len(a)] if x < 0  else a
        z = sign * int(y[len(y)-1:0:-1] + y[0])
        if z < -2147483648 or z > 2147483647:
            return 0
        return  z
sol = Solution()
print sol.reverse(1534236469)
