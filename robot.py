""" Class that defines robot
    left axis goes in minus direction compared to direction of robot
"""
 
from Pololu import ServoController  #impot pololu servo controller interface class

class Robot:
    """ class that represent my robot and his funstions """
    
    def __init__(self):
        """ robot init function """
        self.left_wheel_channel = 2 #set left axis ID on pololu
        self.right_wheel_channel = 1 #set ight axis ID on pololu
        self.null_speed = 1500  #set pont at which drives are stopped, middle of range
        self.max_speed = 1000   #max speed in range units of pololu interface (0-1500)
        self.turn_speed = 10    #preset turn speed in percents
        self.servo_interface = ServoController()
        
    def close_servo_interface(self):
        """ close servo interface to free system resources """
        self.servo_interface.closeServo()
        
    def go_forward(self, speed = 50):
        """ command robot to go foward """
        if speed > 0:
            command_speed = speed*self.max_speed/100 #speed*max_speed/100[percents]
            self.servo_interface.setPosition(self.left_wheel_channel,self.null_speed - command_speed)
            self.servo_interface.setPosition(self.right_wheel_channel,self.null_speed + command_speed)
        else:
            print("Wrong speed set point.\n")
    
    def go_backward(self, speed = 50):
        """ command robot to go backward """
        if speed > 0:
            command_speed = speed*self.max_speed/100 #speed*max_speed/100[percents]
            self.servo_interface.setPosition(self.left_wheel_channel,self.null_speed + command_speed)
            self.servo_interface.setPosition(self.right_wheel_channel,self.null_speed - command_speed)
        else:
            print("Wrong speed set point.\n")
        
    def turn_left(self,speed = 50, turn_speed = 80):
        """ command robot to turn left """
        if speed > 0:
            command_speed = speed*self.max_speed/100 #speed*max_speed/100[percents]
            
            if turn_speed <= 0 or turn_speed > 100: #make sure that turn speed is not out of limits
                calculate_trun_speed = self.turn_speed
            else:
                calculate_trun_speed = turn_speed
                
            speed_for_turn = speed*(100-calculate_trun_speed)/100
            command_turn_speed = speed_for_turn*self.max_speed/100 #speed*max_speed/100[percents]
            
            print("Command speed: " + str(self.null_speed + command_speed))
            print("Command turn speed: " + str(self.null_speed - command_turn_speed))
            
            self.servo_interface.setPosition(self.left_wheel_channel,self.null_speed - command_turn_speed)
            self.servo_interface.setPosition(self.right_wheel_channel,self.null_speed + command_speed)
        else:
            print("Wrong speed set point.\n")
        
    def turn_right(self,speed = 50, turn_speed = 80):
        """ command robot to tun right """
        if speed > 0:
            command_speed = speed*self.max_speed/100 #speed*max_speed/100[percents]
            
            if turn_speed <= 0 or turn_speed > 100: #make sure that turn speed is not out of limits
                calculate_trun_speed = self.turn_speed
            else:
                calculate_trun_speed = turn_speed
                
            speed_for_turn = speed*(100-calculate_trun_speed)/100
            command_turn_speed = speed_for_turn*self.max_speed/100 #speed*max_speed/100[percents]
            
            print("Command speed: " + str(self.null_speed + command_speed))
            print("Command turn speed: " + str(self.null_speed - command_turn_speed))
            
            self.servo_interface.setPosition(self.left_wheel_channel,self.null_speed - command_speed)
            self.servo_interface.setPosition(self.right_wheel_channel,self.null_speed + command_turn_speed)
        else:
            print("Wrong speed set point.\n")
        
    def stop_robot(self):
        """ command robot to stop """
        self.servo_interface.setPosition(self.left_wheel_channel,self.null_speed)
        self.servo_interface.setPosition(self.right_wheel_channel,self.null_speed)
