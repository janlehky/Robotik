import RPi.GPIO as GPIO                    #Import GPIO library
import time                                #Import time library
GPIO.setmode(GPIO.BCM)                     #Set GPIO pin numbering

class Distance():
    """Clas which will run in separate thread and check distance of object from front of robot"""

    def __init__(self):
        self.TRIG = 20
        self.ECHO = 21

        GPIO.setup(self.TRIG, GPIO.OUT)  # Set pin as GPIO out
        GPIO.setup(self.ECHO, GPIO.IN)  # Set pin as GPIO in

    def measure(self, data):
        GPIO.output(self.TRIG, False)  # Set TRIG as LOW
        # print('Waitng For Sensor To Settle')
        time.sleep(2)  # Delay of 2 seconds

        GPIO.output(self.TRIG, True)  # Set TRIG as HIGH
        time.sleep(0.00001)  # Delay of 0.00001 seconds
        GPIO.output(self.TRIG, False)  # Set TRIG as LOW

        while GPIO.input(self.ECHO) == 0:  # Check whether the ECHO is LOW
            pulse_start = time.time()  # Saves the last known time of LOW pulse

        while GPIO.input(self.ECHO) == 1:  # Check whether the ECHO is HIGH
            pulse_end = time.time()  # Saves the last known time of HIGH pulse

        pulse_duration = pulse_end - pulse_start  # Get pulse duration to a variable

        distance = pulse_duration * 17150  # Multiply pulse duration by 17150 to get distance
        distance = round(distance, 2)  # Round to two decimal points

        front_distance = distance - 0.5
        data.value = front_distance
