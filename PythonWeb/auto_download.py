# -*- coding: utf-8 -*-
"""
Created on Wed May 03 10:29:05 2017

@author: alin
"""

# Simple file to download some spanish mp3 from sbs website

import urllib
import urllib2
import re
from bs4 import BeautifulSoup 
import os

url = 'http://www.sbs.com.au/podcasts/yourlanguage/spanish/'
html = urllib.urlopen(url).read()
soup = BeautifulSoup(html)

store = []
for link in soup.find_all('a'):
    address = link.get('href')
    if re.findall('\.mp3', address):
        store.append(address)
        
folder = "/home/alin/SpanishMp3/"
file_in_folder = os.listdir(folder)
for link in store:
    filename = re.findall('spanish_\d+_\d+\.mp3', link)[0]
    print filename 
    if filename not in file_in_folder:
        mp3file = urllib2.urlopen(link)
        to_file = folder + filename
        with open(to_file, "wb") as output:
            output.write(mp3file.read())
        
