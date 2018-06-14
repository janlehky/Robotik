from flask import Flask
from flask import render_template
from flask import request
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import control.motor_control as mc


app = Flask(__name__)
car = mc.MotorControl()
car.configure_gpio()


@app.route('/')
def hello_world():
    # return 'Hello, World!'
    return render_template('slider.html')


@app.route('/slider/')
@app.route('/slider/<name>')
def hello(name=None):
    return render_template('slider.html', name=name)


@app.route('/valueofstearing')
def slide():
    steering_angle = request.args.get('angle')
    print(steering_angle)
    car.set_steering(steering_angle)
    car.refresh_controls()
    return 'Ok'


@app.route('/valueofspeed')
def speed_slider():
    speed = request.args.get('speed')
    print(speed)
    car.set_speed(speed)
    car.refresh_controls()
    return 'Ok'


if __name__ == "__main__":
    try:
        app.run()
    finally:
        car.release_gpio()
