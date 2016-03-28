#code for server side of robot
#code needs update for Python 3, it gets wrong data from socket
import socket
import sys
import threading
from robot import Robot
import RPi.GPIO as GPIO                    #Import GPIO library
import time                                #Import time library
GPIO.setmode(GPIO.BCM)                     #Set GPIO pin numbering 
 
HOST = ''   # Symbolic name meaning all available interfaces
PORT = 8898 # Arbitrary non-privileged port

front_distance = 50     #distance of object from front of robot
drive_command = 0	#drive command defining in which direction robot will go

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
 
#Bind socket to local host and port
try:
    s.bind((HOST, PORT))
except socket.error:
    print('Bind failed')
    sys.exit()
     
print('Socket bind complete')

#create robot class intance
robot = Robot()
print('Robot class created.')
 
#Start listening on socket
s.listen(10)
print('Socket now listening')

class clientThread(threading.Thread):
    """Thread to accept commands from clients"""
    def __init__(self, connection):
        threading.Thread.__init__(self)
        self.connection = connection
    
    def run(self):
        #conn.send('Welcome to the server. Type something and hit enter\n') #send only takes string
        global drive_command #define drive command as global
     
        #infinite loop so that function do not terminate and thread do not end.
        while True:  
            #Receiving from client
            data = self.connection.recv(1024)

            #get command from client
            str_data = str(data)
            if str_data == 'Go Forward':
                print('Driving forward.')
                drive_command = 8	#robot.go_forward()
            elif str_data == 'Go Backward':
                print("Driving backward.")
                drive_command = 2	#robot.go_backward()
            elif str_data == 'Turn Left':
                print("Driving left.")
                drive_command = 4       #robot.turn_left()
            elif str_data == 'Turn Right':
                print("Driving right.")
                drive_command = 6	#robot.turn_right()
            elif str_data == 'Stop':
                print('Robot stopped.')
                drive_command = 5	#robot.stop_robot()
            else:
                print('Unknown command')
                
            #reply = 'OK...' + data
            print(data)
            if not data: 
                break
     
        #came out of loop
        conn.close()

class distanceThread(threading.Thread):
    """Clas which will run in separate thread and check distance of object from front of robot"""
    def __init__(self):
        threading.Thread.__init__(self)
        self.TRIG = 20
        self.ECHO = 21
        
        GPIO.setup(self.TRIG,GPIO.OUT)                  #Set pin as GPIO out
        GPIO.setup(self.ECHO,GPIO.IN)                   #Set pin as GPIO in
        
    def run(self):
        global front_distance #define distance as global
        while True:
            GPIO.output(self.TRIG, False)                 #Set TRIG as LOW
            print('Waitng For Sensor To Settle')
            time.sleep(3)                            #Delay of 2 seconds

            GPIO.output(self.TRIG, True)                  #Set TRIG as HIGH
            time.sleep(0.00001)                      #Delay of 0.00001 seconds
            GPIO.output(self.TRIG, False)                 #Set TRIG as LOW

            while GPIO.input(self.ECHO)==0:               #Check whether the ECHO is LOW
                pulse_start = time.time()              #Saves the last known time of LOW pulse

            while GPIO.input(self.ECHO)==1:               #Check whether the ECHO is HIGH
                pulse_end = time.time()                #Saves the last known time of HIGH pulse 

            pulse_duration = pulse_end - pulse_start #Get pulse duration to a variable

            distance = pulse_duration * 17150        #Multiply pulse duration by 17150 to get distance
            distance = round(distance, 2)            #Round to two decimal points

            if distance > 2 and distance < 400:      #Check whether the distance is within range
                front_distance = distance - 0.5
                print('Distance:' + str(distance - 0.5) + 'cm')  #Print distance with 0.5 cm calibration
            else:
                print('Out Of Range')                   #display out of range

class driverThread(threading.Thread):
	def __init__(self):
            threading.Thread.__init__(self)
            self.old_drive_command = 0
		
	def run(self):
            
            while True:
                time.sleep(2)   #driver thread update time
                print('Drive command: ' + str(drive_command) + ' old ' + str(self.old_drive_command) + ' distance ' + str(front_distance))
                if drive_command != self.old_drive_command:     #if driver gets new command it is executed, otherwise robot keep going
                    self.old_drive_command = drive_command
                    if self.old_drive_command == 8:
                        robot.go_forward()
                    elif self.old_drive_command == 2:
                        robot.go_backward()
                    elif self.old_drive_command == 4:
                        robot.turn_left()
                    elif self.old_drive_command == 6:
                        robot.turn_right()
                    elif self.old_drive_command == 5:
                        robot.stop_robot()
                    else:
                        robot.stop_robot()
			
                if front_distance < 10:
                    robot.stop_robot()
                    self.old_drive_command = 5  #set stop command into memory so robot can be easily restarted
                    print('Obstacle too close')			
		
print('Starting measurement thread')
measurement = distanceThread()
measurement.start()

print('Starting driver')
driver = driverThread()
driver.start()

#now keep talking with the client
while True:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
     
    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    connection_thread = clientThread(conn)
    connection_thread.start()
 
s.close()

#close servo controler
robot.close_servo_interface()
