#Main robot code

print("Welcome to robot interface.\n")

direction = 99

#accept user input for robot control
while True:	
	direction = int(input("Select robot direction:"))
	if direction == 2:
		print("Driving backward.\n")
	elif direction == 4:
		print("Driving left.\n")
	elif direction == 5:
		print('Robot stopped.\n')
	elif direction == 6:
		print("Driving right.\n")
	elif direction == 8:
		print('Driving forward.\n')
	elif direction == 0:
		break
	else:
		print('Not valid command.\n')

print('Finishing program')
