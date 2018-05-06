import Adafruit_PCA9685
import multiprocessing
import RPi.GPIO as GPIO     #Import GPIO library
from Distance import Distance
import paho.mqtt.client as mqtt


GPIO.setmode(GPIO.BCM)  # Set GPIO pin numbering

# define values for communication between processes
x_offset = multiprocessing.Value('d', 0.0)
y_offset = multiprocessing.Value('d', 0.0)
run_cmd = multiprocessing.Value('d', 0.0)
front_distance_sensor = multiprocessing.Value('d', 0.0)

max_speed = 100     # Maximum speed of vehicle


def drive_vehicle(x_offset, y_offset, run_cmd, front_distance_sensor):
    """Logic for controlling car movement"""
    # Initialise the PCA9685 using the default address (0x40).
    pwm = Adafruit_PCA9685.PCA9685()

    # set number of pins for direction of drives
    left_fwd_pin = 4
    left_bwd_pin = 17

    right_fwd_pin = 22
    right_bwd_pin = 27

    GPIO.setup(left_fwd_pin, GPIO.OUT)  # left forward pin
    GPIO.setup(left_bwd_pin, GPIO.OUT)  # left backward pin

    GPIO.setup(right_fwd_pin, GPIO.OUT)  # right forward pin
    GPIO.setup(right_bwd_pin, GPIO.OUT)  # right backward pin

    GPIO.output(right_fwd_pin, True)
    GPIO.output(right_bwd_pin, False)

    GPIO.output(right_fwd_pin, True)
    GPIO.output(right_bwd_pin, False)

    while True:
        # print('X_offset: {}'.format(x_offset.value))
        if front_distance_sensor.value < 5:
            left_speed = 0
            right_speed = 0
        else:
            if x_offset.value == -10:
                left_speed = 0
                right_speed = 0.3 * max_speed
            elif -5 < x_offset.value < 0:
                left_speed = abs(x_offset.value) * max_speed
                right_speed = max_speed
            elif x_offset.value > 0:
                left_speed = max_speed
                right_speed = x_offset.value * max_speed
            else:
                left_speed = max_speed
                right_speed = max_speed
        
        print('Speeds: Left {} Right {}'.format(left_speed, right_speed))

        # Right drives
        pwm.set_pwm(0, 0, int(right_speed*run_cmd.value))
        pwm.set_pwm(1, 0, int(right_speed*run_cmd.value))

        # Left drives
        pwm.set_pwm(4, 0, int(left_speed*run_cmd.value))
        pwm.set_pwm(5, 0, int(left_speed*run_cmd.value))


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("demo.key")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    text = str(msg.payload)
    data = text.split(";")
    if 'go' in data[0]:
        parts = data[1].split(':')
        run_cmd.value = 1.0
        if 'x' in parts[0]:
            print('X offset: {}'.format(parts[1]))
            x_offset.value = float(parts[1][:-1])
        elif 'y' in parts[0]:
            print('Y offset: {}'.format(parts[1]))
            y_offset.value = float(parts[1][:-1])
    elif 'stop' in data[0]:
        run_cmd.value = 0.0


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
rabitmq_ip = '192.168.2.88'
client.username_pw_set('client', '1234')
client.connect(rabitmq_ip, 1883, 60)

# Create distance measurement process and start it
drive_process = multiprocessing.Process(target=drive_vehicle,
                                        args=(x_offset, y_offset, run_cmd, front_distance_sensor, ))
drive_process.start()

# Create measurement object for front sensor
front_measurement = Distance(20, 21)
# Create front measurement process and start it
front_measurement_process = multiprocessing.Process(target=front_measurement.measure, args=(front_distance_sensor, ))
front_measurement_process.start()

try:
    client.loop_forever()
finally:
    # Terminate running processes and cleanup claimed GPIO's
    drive_process.terminate()
    front_measurement_process.terminate()
    front_measurement.cleanup()
    GPIO.cleanup()
