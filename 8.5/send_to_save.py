import socket

def send_command():
    msgFromClient       = bytes.fromhex('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
    #bytesToSend         = str.encode(msgFromClient)
    serverAddressPort   = ("192.168.0.2", 8080)
    bufferSize          = 18

    print(msgFromClient)
    # Create a UDP socket at client side
    UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    UDPClientSocket.sendto(msgFromClient, serverAddressPort)

    #msgFromServer = UDPClientSocket.recvfrom(bufferSize)
    #msg = "Message from Server {}".format(msgFromServer[0])
    #print(msg)

if __name__ == '__main__':
	send_command()