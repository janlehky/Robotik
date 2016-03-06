#Main robot code
from robot import Robot

print("Welcome to robot interface.\n")

direction = 99

#create robot class intance
robot = Robot()

#accept user input for robot control
while True:	
	direction = int(input("Select robot direction:"))
	if direction == 2:
		print("Driving backward.\n")
		robot.go_backward()
	elif direction == 4:
		print("Driving left.\n")
		robot.turn_left()
	elif direction == 5:
		print('Robot stopped.\n')
		robot.stop_robot()
	elif direction == 6:
		print("Driving right.\n")
		robot.turn_right()
	elif direction == 8:
		print('Driving forward.\n')
		robot.go_forward()
	elif direction == 0:
		break
	else:
		print('Not valid command.\n')

print('Finishing program.')

#close servo controler
robot.close_servo_interface()
