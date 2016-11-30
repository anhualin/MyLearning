#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      alin
#
# Created:     17/11/2016
# Copyright:   (c) alin 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------



import urllib
import re
from BeautifulSoup import *

url = 'http://python-data.dr-chuck.net/comments_333990.html'
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
