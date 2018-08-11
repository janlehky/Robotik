import paho.mqtt.publish as publish
import time

rabitmq_ip = '192.168.2.88'

user_auth = {'username': 'client', 'password': '1234'}

msgs = [{'topic': "demo.key", 'payload': 'x:{}'.format(32.3)},
        {'topic': 'demo.key', 'payload': 'y:{}'.format(12.1)}]
publish.multiple(msgs, rabitmq_ip, auth=user_auth)
time.sleep(5)
