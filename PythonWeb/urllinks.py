# Note - this code must run in Python 2.x and you must download
# http://www.pythonlearn.com/code/BeautifulSoup.py
# Into the same folder as this program

import urllib
from BeautifulSoup import *
import re


url = 'http://python-data.dr-chuck.net/known_by_Kiaya.html'
##html = urllib.urlopen(url).read()
##soup = BeautifulSoup(html)


def followLink(url, pos, count):
    for i in range(0, count):
        html = urllib.urlopen(url).read()
        soup = BeautifulSoup(html)
        tags = soup('a')
        tag = tags[pos - 1]
        url = tag.get('href', None)
    print url
    names = re.findall('by_(\w+)\.html', url)
    print names[0]

followLink(url, 18, 7)