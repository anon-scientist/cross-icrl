"""
Used to connect to a running RLAgent instance, for sending command
strings for debugging. This tool just send single strings, one at a time.
Interpretation is up to RLAgent...
By default we send onj port 11000, another one can be sepcified as first cmd line argument
"""

import socket, sys ;
TCP_IP = '127.0.0.1'
TCP_PORT = 11000 ;
if len(sys.argv) > 1:
  TCP_PORT = int(sys.argv[1]) ;
BUFFER_SIZE = 1024
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

while True:
  print("----------------") ;
  msg = input("What do we send?") ;
  s.sendall(msg.encode()) ;
  print("Sent request") ;
  data = s.recv(BUFFER_SIZE)
  print ("received confirmation:", data)
