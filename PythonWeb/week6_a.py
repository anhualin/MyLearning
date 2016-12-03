import urllib
import json

url = raw_input('Enter url: ')
#url = 'http://python-data.dr-chuck.net/comments_42.json'
#url = 'http://python-data.dr-chuck.net/comments_333991.json'

print 'Retrieving', url
uh = urllib.urlopen(url)
data = uh.read()
print 'Retrieved',len(data),'characters'
print data


try: js = json.loads(str(data))
except: js = None
if 'status' not in js or js['status'] != 'OK':
    print '==== Failure To Retrieve ===='
    print data


print json.dumps(js, indent=4)

comments = js['comments']
total = sum([e['count'] for e in comments])
print "total = ", total