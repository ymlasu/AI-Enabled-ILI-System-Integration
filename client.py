'''
Client that communicates with the raspberry pi microcontroller to receive data from stepper motor encoder.
'''

import urllib.request
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--ip", help = 'Raspberry Pi Addr')
args = parser.parse_args()

ip = args.ip
link = 'http://' + ip + ':8080'
try:
	while True:
		start_time = time.time()
		f = urllib.request.urlopen(link)
		myfile = f.read()
		print("Orig: ", myfile, "Convert:", float(myfile))
		myfile.split()
		print("--- %s Hz ---" % (1/(time.time() - start_time)))

except:
	exit()
