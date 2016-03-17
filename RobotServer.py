#code for server side of robot
import socket
import sys
import threading
 
HOST = ''   # Symbolic name meaning all available interfaces
PORT = 8888 # Arbitrary non-privileged port
 
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
 
#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error , msg:
    print('Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1])
    sys.exit()
     
print('Socket bind complete')
 
#Start listening on socket
s.listen(10)
print('Socket now listening')

class clientThread(threading.Thread):
    """Thread to accept commands from clients"""
    def __init__(self, connection)
        threading.Thread.__init__(self)
        self.connection = connection
    
    def run(self):
        conn.send('Welcome to the server. Type something and hit enter\n') #send only takes string
     
        #infinite loop so that function do not terminate and thread do not end.
        while True:
         
            #Receiving from client
            data = conn.recv(1024)
            reply = 'OK...' + data
            if not data: 
                break
     
            conn.sendall(reply)
     
        #came out of loop
        conn.close()

#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
     
    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    connection_thread = clientThread(conn)
 
s.close()
