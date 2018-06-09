from flask import Flask
from flask import render_template
from flask import request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/slider/')
@app.route('/slider/<name>')
def hello(name=None):
    return render_template('slider.html', name=name)


@app.route('/valueofslider')
def slide():
    a = request.args.get('a')
    print(a)
    return 'Ok'
