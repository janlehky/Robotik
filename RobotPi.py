#Main robot code
from Pololu import ServoController

print("Welcome to robot interface.\n")

direction = 99

#create instance of servo class and open channel
servo_drive = ServoController()

#accept user input for robot control
while True:	
	direction = int(input("Select robot direction:"))
	if direction == 2:
		print("Driving backward.\n")
	elif direction == 4:
		print("Driving left.\n")
	elif direction == 5:
		print('Robot stopped.\n')
		servo_drive.setPosition(1,1500)
		servo_drive.setPosition(2,1500)
	elif direction == 6:
		print("Driving right.\n")
	elif direction == 8:
		print('Driving forward.\n')
		servo_drive.setPosition(1,500)
		servo_drive.setPosition(2,500)
	elif direction == 0:
		break
	else:
		print('Not valid command.\n')

print('Finishing program')

#close servo controler
servo_drive.closeServo()
