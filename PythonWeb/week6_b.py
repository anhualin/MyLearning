import urllib
import json

# serviceurl = 'http://maps.googleapis.com/maps/api/geocode/json?'
serviceurl = 'http://python-data.dr-chuck.net/geojson?'


address = raw_input('Enter location: ')
#address = 'South Federal University'
#address = 'Universidade Tecnica de Lisboa'

url = serviceurl + urllib.urlencode({'sensor':'false', 'address': address})
print 'Retrieving', url
uh = urllib.urlopen(url)
data = uh.read()
print 'Retrieved',len(data),'characters'

try: js = json.loads(str(data))
except: js = None
if 'status' not in js or js['status'] != 'OK':
    print '==== Failure To Retrieve ===='
    print data
 

#print json.dumps(js, indent=4)
place_id = js['results'][0]['place_id']
print 'place_id = ', place_id


