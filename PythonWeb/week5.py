#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:36:47 2016

@author: alin
"""

import urllib
import xml.etree.ElementTree as ET


url = raw_input('Enter url: ')
#url = 'http://python-data.dr-chuck.net/comments_333987.xml'

print 'Retrieving', url
uh = urllib.urlopen(url)
data = uh.read()
print 'Retrieved',len(data),'characters'
print data
tree = ET.fromstring(data)

counts = tree.findall('.//count')
total = 0
for cnt in counts:
    total += int(cnt.text)
    

print 'total = ', total
