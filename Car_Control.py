import Adafruit_PCA9685
import multiprocessing
import RPi.GPIO as GPIO     #Import GPIO library
from Distance import Distance
import paho.mqtt.client as mqtt


GPIO.setmode(GPIO.BCM)  # Set GPIO pin numbering

# define values for communication between processes
x_offset = multiprocessing.Value('d', 0.0)
y_offset = multiprocessing.Value('d', 0.0)
width = multiprocessing.Value('d', 0.0)
run_cmd = multiprocessing.Value('d', 0.0)
front_distance_sensor_1 = multiprocessing.Value('d', 10.0)
front_distance_sensor_2 = multiprocessing.Value('d', 10.0)

max_speed = 4000     # Maximum speed of vehicle


def drive_vehicle(x_offset, y_offset, run_cmd, width, front_distance_sensor_1, front_distance_sensor_2):
    """Logic for controlling car movement"""
    # Initialise the PCA9685 using the default address (0x40).
    pwm = Adafruit_PCA9685.PCA9685()

    # set number of pins for direction of drives
    left_fwd_pin_1 = 4
    left_fwd_pin_2 = 17
    left_bwd_pin_1 = 18
    left_bwd_pin_2 = 23

    right_fwd_pin_1 = 22
    right_fwd_pin_2 = 27
    right_bwd_pin_1 = 24
    right_bwd_pin_2 = 25

    GPIO.setup(left_fwd_pin_1, GPIO.OUT)  # left forward 1 pin
    GPIO.setup(left_fwd_pin_2, GPIO.OUT)  # left forward 2 pin
    GPIO.setup(left_bwd_pin_1, GPIO.OUT)  # left backward 1 pin
    GPIO.setup(left_bwd_pin_2, GPIO.OUT)  # left backward 2 pin

    GPIO.setup(right_fwd_pin_1, GPIO.OUT)  # right forward 1 pin
    GPIO.setup(right_fwd_pin_2, GPIO.OUT)  # right forward 2 pin
    GPIO.setup(right_bwd_pin_1, GPIO.OUT)  # right backward 1 pin
    GPIO.setup(right_bwd_pin_2, GPIO.OUT)  # right backward 2 pin

    left_fwd = True
    left_bwd = False

    right_fwd = True
    right_bwd = False

    while True:
        try:
            # Take shortest distance measured by ultrasound
            if front_distance_sensor_1.value < front_distance_sensor_2.value:
                front_distance = front_distance_sensor_1.value
            else:
                front_distance = front_distance_sensor_2.value

            if front_distance < 5 or width.value > 500:
                # if we are facing some obstacle or object we are looking for is close > stop
                left_speed = 0
                right_speed = 0
                left_fwd = left_bwd = right_fwd = right_bwd = False
            else:
                right_fwd = True
                if x_offset.value == -10:
                    # no object is detected by camera
                    left_speed = 0.6 * max_speed
                    left_fwd = False
                    left_bwd = True
                    right_speed = 0.8 * max_speed
                elif -5 < x_offset.value < 0:
                    # object is in left part of the screen
                    left_speed = pow(abs(x_offset.value), 2) * max_speed
                    right_speed = max_speed
                    left_fwd = True
                    left_bwd = False
                elif x_offset.value > 0:
                    # object is in right part of the screen
                    left_speed = max_speed
                    right_speed = pow(x_offset.value, 2) * max_speed
                    left_fwd = True
                    left_bwd = False
                else:
                    # object is in the middle
                    left_speed = max_speed
                    right_speed = max_speed
                    left_fwd = True
                    left_bwd = False
        
            print('Speeds: Left {} Right {} Run {}'.format(left_speed, right_speed, run_cmd.value))

            # Right drives
            pwm.set_pwm(0, 0, int(right_speed*run_cmd.value))
            pwm.set_pwm(1, 0, int(right_speed*run_cmd.value))
            GPIO.output(left_fwd_pin_1, left_fwd)
            GPIO.output(left_fwd_pin_2, left_fwd)
            GPIO.output(left_bwd_pin_1, left_bwd)
            GPIO.output(left_bwd_pin_2, left_bwd)

            # Left drives
            pwm.set_pwm(4, 0, int(left_speed*run_cmd.value))
            pwm.set_pwm(5, 0, int(left_speed*run_cmd.value))
            GPIO.output(right_fwd_pin_1, right_fwd)
            GPIO.output(right_fwd_pin_2, right_fwd)
            GPIO.output(right_bwd_pin_1, right_bwd)
            GPIO.output(right_bwd_pin_2, right_bwd)
        except KeyboardInterrupt:
            # Stop robot after keyboard interrupt
            GPIO.output(left_fwd_pin_1, False)
            GPIO.output(left_fwd_pin_2, False)
            GPIO.output(left_bwd_pin_1, False)
            GPIO.output(left_bwd_pin_2, False)
            GPIO.output(right_fwd_pin_1, False)
            GPIO.output(right_fwd_pin_2, False)
            GPIO.output(right_bwd_pin_1, False)
            GPIO.output(right_bwd_pin_2, False)
            GPIO.cleanup()


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
        run_cmd.value = 1.0
        for item in data[1:]:
            parts = item.split(':')
            if 'x' in parts[0]:
                print('X offset: {}'.format(parts[1]))
                x_offset.value = float(parts[1])
            elif 'width' in parts[0]:
                print('Width: {}'.format(parts[1]))
                width.value = float(parts[1][:-1])
            elif 'y' in parts[0]:
                print('Y offset: {}'.format(parts[1]))
                y_offset.value = float(parts[1])
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
                                        args=(x_offset, y_offset, run_cmd, width, front_distance_sensor_1,
                                              front_distance_sensor_2, ))
drive_process.start()

# Create measurement object for front sensor
front_measurement_1 = Distance(20, 21)
front_measurement_2 = Distance(26, 19)
# Create front measurement process and start it
front_measurement_process_1 = multiprocessing.Process(target=front_measurement_1.measure,
                                                      args=(front_distance_sensor_1, ))
front_measurement_process_1.start()

front_measurement_process_2 = multiprocessing.Process(target=front_measurement_2.measure,
                                                      args=(front_distance_sensor_2, ))
front_measurement_process_2.start()

try:
    client.loop_forever()
finally:
    # disconnect from rabitmq
    print("Disconnect")
    client.disconnect()
    # Terminate running processes and cleanup claimed GPIO's
    drive_process.terminate()
    front_measurement_process_1.terminate()
    front_measurement_1.cleanup()
    front_measurement_process_2.terminate()
    front_measurement_2.cleanup()
