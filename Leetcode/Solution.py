#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 15:00:52 2017

@author: alin
"""

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        len1 = len(nums1)
        len2 = len(nums2)
        if (len1 + len2) % 2 == 0:
            half = (len1 + len2 - 2) / 2
            l1 = 0
            u1 = len1 - 1
            l2 = 0
            u2 = len2 - 1
           
            x = int((l1 + u1)/2)
            y = int((l2 + u2)/2)
            print 'x=', x
            print 'y=', y
            # easy case
            # general cases
            while(u1 - l1 >= 2 and u2 - l2 >= 2):
                print nums1[x], nums2[y], x, y, half
                if nums1[x] >= nums2[y]:
                    if x + y >= half:
                        u1 = x
                        x = int((l1 + u1)/2)
                    else:
                        l2 = y
                        y = int((l2 + u2)/2)                           
                else:
                    if x + y >= half:
                        u2 = y
                        y = int((l2 + u2)/2)
                    else:
                        l1 = x
                        x = int((l1 + u1)/2)
            print l1, u1, l2, u2
            print nums1[l1], nums1[u1]
            print nums2[l2], nums2[u2]

a = Solution()
a.findMedianSortedArrays([1,1,1,1], [2,2,2,2,2,2])    
