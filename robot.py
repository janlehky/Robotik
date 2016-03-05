""" Class that defines robot """
from Pololu import ServoController  #impot pololu servo controller interface class

class Robot():
    """ class that represent my robot and his funstions """
    
    def ___init___(self):
        """ robot init function """
        self.left_wheel_channel = 1 #set left axis ID on pololu
        self.ight_wheel_channel = 2 #set ight axis ID on pololu
        self.servo_interface = ServoController()
        
    def close_servo_interface(self):
        """ close servo interface to free system resources """
        self.servo_interface.closeServo()
        
    def go_forward(self, speed = 50):
        """ command robot to go foward """
        self.servo_interface.setPosition(self.left_wheel_channel,2000)
        self.servo_interface.setPosition(self.right_wheel_channel,2000)
    
    def go_backward(self, speed = 50):
        """ command robot to go backward """
        self.servo_interface.setPosition(self.left_wheel_channel,1000)
        self.servo_interface.setPosition(self.right_wheel_channel,1000)
        
    def turn_left(self,speed = 50):
        """ command robot to turn left """
        self.servo_interface.setPosition(self.left_wheel_channel,1000)
        self.servo_interface.setPosition(self.right_wheel_channel,2000)
        
    def turn_right(self,speed = 50):
        """ command robot to tun right """
        self.servo_interface.setPosition(self.left_wheel_channel,2000)
        self.servo_interface.setPosition(self.right_wheel_channel,1000)
        
    def stop_robot(self):
        """ command robot to stop """
        self.servo_interface.setPosition(self.left_wheel_channel,1500)
        self.servo_interface.setPosition(self.right_wheel_channel,1500)
