import Adafruit_PCA9685
import RPi.GPIO as GPIO     #Import GPIO library


class MotorControl:
    """Provide access to control function of car, controls speed, stearing"""

    def __init__(self):
        GPIO.setmode(GPIO.BCM)

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

        # set base speed values
        self.speed = 0
        self.right_speed = 0
        self.left_speed = 0
        self.max_speed = 4000

        # set base steering angle
        self.steering = 0

        # set initial directions for every drive
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

    def set_speed(self, speed_in):
        """Sets actual value of speed and recalculates speeds (calling calculate speeds method)"""
        if int(speed_in) < 0:
            self.left_fwd = False
            self.left_bwd = True
            self.right_fwd = False
            self.right_bwd = True

            self.speed = - int(speed_in)
        else:
            self.left_fwd = True
            self.left_bwd = False
            self.right_fwd = True
            self.right_bwd = False

            self.speed = int(speed_in)

        self.calculate_speeds()

    def set_steering(self, steering_angle):
        """Sets actual value of steering and recalculates speeds (calling calculate speeds method)"""
        self.steering = int(steering_angle)

        self.calculate_speeds()

    def calculate_speeds(self):
        """This method recalculates left and right speed based on actually set speed and steering"""
        steer_factor = self.steering / 100
        print("Calculated factor: {}".format(steer_factor))
        if steer_factor < 0:
            self.left_speed = (1 + steer_factor) * self.max_speed * self.speed / 100
            self.right_speed = self.max_speed * self.speed / 100
        elif steer_factor == 0:
            self.left_speed = self.max_speed * self.speed / 100
            self.right_speed = self.max_speed * self.speed / 100
        else:
            self.left_speed = self.max_speed * self.speed / 100
            self.right_speed = (1 - steer_factor) * self.max_speed * self.speed / 100
            print("left: {} right: {}".format(self.left_speed, self.right_speed))

    def refresh_controls(self):
        """Refresh actual commands to every channel based on set speed and steering"""
        # Right drives
        self.pwm.set_pwm(0, 0, int(self.right_speed))
        self.pwm.set_pwm(1, 0, int(self.right_speed))
        GPIO.output(self.left_fwd_pin_1, self.left_fwd)
        GPIO.output(self.left_fwd_pin_2, self.left_fwd)
        GPIO.output(self.left_bwd_pin_1, self.left_bwd)
        GPIO.output(self.left_bwd_pin_2, self.left_bwd)

        # Left drives
        self.pwm.set_pwm(4, 0, int(self.left_speed))
        self.pwm.set_pwm(5, 0, int(self.left_speed))
        GPIO.output(self.right_fwd_pin_1, self.right_fwd)
        GPIO.output(self.right_fwd_pin_2, self.right_fwd)
        GPIO.output(self.right_bwd_pin_1, self.right_bwd)
        GPIO.output(self.right_bwd_pin_2, self.right_bwd)
