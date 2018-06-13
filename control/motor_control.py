import Adafruit_PCA9685
import RPi.GPIO as GPIO     #Import GPIO library


class MotorControl:
    """Provide access to control function of car, controls speed, stearing"""

    def __init__(self):
        # Initialise the PCA9685 using the default address (0x40).
        self.pwm = Adafruit_PCA9685.PCA9685()

        # set number of pins for direction of drives
        self.left_fwd_pin_1 = 4
        self.left_fwd_pin_2 = 17
        self.left_bwd_pin_1 = 18
        self.left_bwd_pin_2 = 23

        self.right_fwd_pin_1 = 22
        self.right_fwd_pin_2 = 27
        self.right_bwd_pin_1 = 24
        self.right_bwd_pin_2 = 25

        self.speed = 0
        self.steering = 0

        self.left_fwd = True
        self.left_bwd = False

        self.right_fwd = True
        self.right_bwd = False

    def configure_gpio(self):
        """Configure gpio before using it"""
        GPIO.setup(self.left_fwd_pin_1, GPIO.OUT)  # left forward 1 pin
        GPIO.setup(self.left_fwd_pin_2, GPIO.OUT)  # left forward 2 pin
        GPIO.setup(self.left_bwd_pin_1, GPIO.OUT)  # left backward 1 pin
        GPIO.setup(self.left_bwd_pin_2, GPIO.OUT)  # left backward 2 pin

        GPIO.setup(self.right_fwd_pin_1, GPIO.OUT)  # right forward 1 pin
        GPIO.setup(self.right_fwd_pin_2, GPIO.OUT)  # right forward 2 pin
        GPIO.setup(self.right_bwd_pin_1, GPIO.OUT)  # right backward 1 pin
        GPIO.setup(self.right_bwd_pin_2, GPIO.OUT)  # right backward 2 pin

        self.left_fwd = True
        self.left_bwd = False

        self.right_fwd = True
        self.right_bwd = False

    def release_gpio(self):
        """Release GPIO when we don't need it"""
        GPIO.cleanup()

    def set_speed(self):
        """Sets actual value of speed"""
        pass

    def set_steering(self):
        """Sets actual value of steering"""
        pass
