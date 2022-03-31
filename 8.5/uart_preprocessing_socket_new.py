import socket
import struct
import sys
#import serial
import time
import signal
from threading import Thread, Lock

class UDP2UART():
    def __init__(self):
        self.PORT_UDP = 51000
        self.FRAME_HEIGHT = 1080
        self.FRAME_WIDTH = 1920
        self.ANGLE_RESOLUTION_X = 0.0507
        self.ANGLE_RESOLUTION_Y = 0.0507
        self.run = False
        #self.buffer
                 
    def signal_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
    
    def establish_connection(self):
        udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        udp_socket.bind(('', self.PORT_UDP))
        print("UDP server up and listening")
        return udp_socket
        
    def receive(self, socket):
        """
        0000   7c 50 79 05 f8 f7 00 04 4b ec 57 a6 08 00 45 00
        0010   00 2e 47 a1 40 00 40 11 71 60 c0 a8 00 02 c0 a8
        0020   00 6b e9 01 cb 20 00 1a 5a 49 aa aa 01 7d 00 01
        0030   04 1b 02 c3 00 2c 00 17 3f 00 7d 46
        """
        buffer_size = 512
        input_bytes = socket.recvfrom(buffer_size)[0]
        #self.buffer = input_bytes
        return input_bytes
    
    def handling_bytes(self, input_bytes):
        frame = int.from_bytes( input_bytes[2:4], "big")
        left = int.from_bytes( input_bytes[6:8], "big")
        top = int.from_bytes( input_bytes[8:10], "big")
        width = int.from_bytes( input_bytes[10:12], "big")
        height = int.from_bytes( input_bytes[12:14], "big")
        conf = float(struct.unpack('>f', input_bytes[14:18])[0])
        
        angle_x = ((left + (width / 2)) - self.FRAME_WIDTH / 2) * self.ANGLE_RESOLUTION_X;
        angle_y = ((self.FRAME_HEIGHT / 2 - (top + (height / 2))) * self.ANGLE_RESOLUTION_Y) - 5;

        angle_x_bytes = bytearray(struct.pack(">f", angle_x))  
        angle_y_bytes = bytearray(struct.pack(">f", angle_y))  

        msg = bytes.fromhex('AA02') + angle_y_bytes + angle_x_bytes
        print("frame {} left {} top {} width {} height {} conf {}".format(frame,left,top,width,height,conf))
        print([ "0x%02x" % b for b in msg ])    
        #UDPServerSocket.close()
        #UDPServerSocket = None
        

if __name__ == '__main__':
    connection = UDP2UART()
    socket = connection.establish_connection()
    
    while True:
        bytes_ = connection.receive(socket)
        connection.handling_bytes(bytes_)
        #print(bytes_)
    
    
    
    #print("\nframe {} left {} top {} width {} height {} conf {}".format(frame,left,top,width,height,conf))
    #print(angle_x, angle_y)
    #print([ "0x%02x" % b for b in angle_x_bytes ])
    #print([ "0x%02x" % b for b in angle_y_bytes ])
    #print([ "0x%02x" % b for b in msg ])

    #print(angle_x_bytes, angle_y_bytes)
    #_thread.start_new_thread(receive,(None,))
    #receive()

    
    