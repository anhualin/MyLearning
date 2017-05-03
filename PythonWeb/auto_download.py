# -*- coding: utf-8 -*-
"""
Created on Wed May 03 10:29:05 2017

@author: alin
"""



import urllib
import re
from bs4 import BeautifulSoup 

url = 'http://www.sbs.com.au/podcasts/yourlanguage/spanish/'
html = urllib.urlopen(url).read()

soup = BeautifulSoup(html)

# Retrieve all of the anchor tags
tags = soup('span')
total = 0
for tag in tags:
    # Look at the parts of a tag
    num = re.findall('(\d+)', str(tag))
    total += int(num[0])
print total
