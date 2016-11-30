# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 20:52:52 2016

@author: alin
"""

import re

directory = ''
directory = 'C:/Users/alin/Documents/SelfStudy/PythonWeb/Week2/'
with open(directory + 'regex_sum_333985.txt', 'r') as myfile:
    data=myfile.read().replace('\n', ' ')

numbers = re.findall('([0-9]+)', data)
total  = sum([int(a) for a in numbers])
print total