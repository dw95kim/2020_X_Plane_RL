# client
import socket
import numpy as np
import os
import struct

recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

send_ip = "localhost"
send_port = 9090
send_addr = (send_ip, send_port)

size = 1024
type_scale = 4

recv_sock.bind(("localhost", 8080))

while True:
    recv_data = [0 for i in range(2)]
    received_data, addr = recv_sock.recvfrom(128)

    for i in range(0, 2):
        x = struct.unpack('f', received_data[0 + type_scale * i : type_scale + type_scale * i])[0]
        print(x)
        recv_data[i] = float(x)
        print(x)
        print(" ")
    
    # send another data
    recv_data[0] += 1

    # send data
    ba = bytearray(struct.pack("f", recv_data[0]))
    for i in range(1, 2):
        ba += bytearray(struct.pack("f", recv_data[i]))
    send_sock.sendto(ba, send_addr)