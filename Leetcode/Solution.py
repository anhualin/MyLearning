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
       
        if len1 == 0:
            if len2 % 2 == 0:
                return float(nums2[len2/2 - 1 ] + nums2[len2/2])/2.0
            else:
                return nums2[len2/2]
        if len2 == 0:
            if len1 % 2 == 0:
                return float(nums1[len1/2 - 1 ] + nums1[len1/2])/2.0
            else:
                return nums1[len1/2]
            
        half = int((len1 + len2) / 2)
        l1 = 0
        u1 = len1 - 1
        l2 = 0
        u2 = len2 - 1
      
        while(u1 - l1 >= 2 and u2 - l2 >= 2):
            x = int((l1 + u1)/2)
            y = int((l2 + u2)/2)
            if nums1[x] >= nums2[y]:
                if x + y >= half - 1:
                    u1 = x
                    x = int((l1 + u1)/2)
                else:
                    l2 = y
                    y = int((l2 + u2)/2)                           
            else:
                if x + y >= half - 1:
                    u2 = y
                    y = int((l2 + u2)/2)
                else:
                    l1 = x
                    x = int((l1 + u1)/2)
        
        if u1 - l1 > 1:
            tmp = nums1
            nums1 = nums2
            nums2 = tmp
            len1 = len(nums1)
            len2 = len(nums2)
            t = l1
            l1 = l2
            l2 = t
            t = u1
            u1 = u2
            u2 = t
        while(u2 - l2 >= 2):
            y = int((l2 + u2)/2)
            if nums2[y] < nums1[l1]:
                if nums1[l1] > nums1[0]:
                    l2 = y
                else:
                    if y >= half:
                        if (len1 + len2) % 2 == 0:
                            return float(nums2[half - 1] + nums2[half])/2.0
                        else:
                            
                            return float(nums2[half])
                    l2 = y
            elif nums2[y] > nums1[u1]:
                if nums1[u1] < nums1[len1 - 1]:
                    u2 = y
                else:
                    if len2 - y - 1 >= half:
                        if (len1 + len2) % 2 == 0:
                            return float(nums2[len2 - half - 1] + nums2[len2 - half])/2.0
                        else:
                          
                            return float(nums2[len2 - half - 1])
                    u2 = y
            elif nums2[y] == nums1[l1]:
                if l1 + y < half - 1:
                    l2 = y
                else:
                    u2 = y
            elif nums2[y] == nums1[u1]:
                if u1 + y < half - 1:
                    l2 = y
                else:
                    u2 = y
            else:
                if l1 + y + 1 >= half:
                    u2 = y
                else:
                    l2 = y
                        
        candidates = set([nums1[l1], nums1[u1], nums2[l2], nums2[u2]])   
       
        chosen = []
        if (len1 + len2) % 2 == 0:
            for c in candidates:
                pos0, pos1 = self.findPos(c, nums1, nums2)
                if pos0 <= half and pos1 >= half + 1:
                    return float(c)
                if (pos0 <= half and pos1 >= half) or (pos0 <= half + 1 and pos1 >= half + 1):
                    chosen.append(c)
                    if len(chosen) == 2:
                        return (float(chosen[0] + chosen[1])) / 2.0
        else:
            for c in candidates:
                pos0, pos1 = self.findPos(c, nums1, nums2)
                if pos0 <= half + 1 and pos1 >= half + 1:
                    return float(c)
               

   
    def findPos(self, x, nums1, nums2):
        pl1 = self.findLower(x, nums1)
        pl2 = self.findLower(x, nums2)
        pu1 = self.findUpper(x, nums1)
        pu2 = self.findUpper(x, nums2)
        pos0 = pl1 + pl2 + 3
        pos1 = pu1 + pu2
        return [pos0, pos1]
    def findLower(self, x,nums):
        N = len(nums)
        if x <= nums[0]:
            return -1
        elif x > nums[N-1]:
            return N-1
        else:
            pl = 0
            pu = N - 1
            while pu - pl > 1:
                z = int((pu + pl)/2)
                if x > nums[z]:
                    pl = z
                else:
                    pu = z
            return pl
    def findUpper(self, x,nums):
        N = len(nums)
        if x < nums[0]:
            return 0
        elif x >= nums[N-1]:
            return N
        else:
            pl = 0
            pu = N - 1
            while pu - pl > 1:
                z = int((pu + pl)/2)
                if x < nums[z]:
                    pu = z
                else:
                    pl = z
            return pu
        
a = Solution()


print a.findMedianSortedArrays([], [1])    


import numpy as np
k = 1
diff = []
while k < 10000:
    n1 =np.random.randint(1,100,1)[0]
    n2 = np.random.randint(1,100,1)[0]
    
    nums1 = np.random.randint(1,100,n1).tolist()
    nums1.sort()
    
    nums2 = np.random.randint(1,100,n2).tolist()
    nums2.sort()
    
    nums = nums1 + nums2
    nums.sort()
    
    if (n1 + n2) %2 ==0:
        m1 = nums[(n1+n2)/2 - 1]
        m2 = nums[(n1+n2)/2]
        r = float(m1 + m2)/2.0
    else:
        i = int((n1 + n2)/2)
        r = float(nums[i])
    result = a.findMedianSortedArrays(nums1, nums2)    
    dif = r - result
    diff.append(dif)
    if dif > 0.1 or dif < -0.1:
        break;
    k = k + 1
