from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)


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
    a = request.args.get('angle')
    print(a)
    return 'Ok'


@app.route('/valueofspeed')
def speed_slider():
    speed = request.args.get('speed')
    print(speed)
    return 'Ok'
