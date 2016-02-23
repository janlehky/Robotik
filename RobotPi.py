#Main robot code

print("Welcome to robot interface.\n")

direction = 99

while direction != 0:
	direction = int(input("Select robot direction:"))
	if direction == 2:
		print("Driving backward.\n")
	elif direction == 4:
		print("Driving left.\n")
	elif direction == 0:
		exit()
