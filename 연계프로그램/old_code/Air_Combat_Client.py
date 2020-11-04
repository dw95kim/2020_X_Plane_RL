import socket
import time

sock = socket.socket( socket.AF_INET , socket.SOCK_DGRAM )
index = 0

while True:
    str_data = str([index, 2, 3])
    sock.sendto(str_data.encode() , ('192.168.0.2',8080) )
    data , addr = sock.recvfrom(200)

    time.sleep(1)
    index += 1
