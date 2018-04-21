import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish
import time


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("demo.key")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
rabitmq_ip = '192.168.2.88'
# rabitmq_ip = "iot.eclipse.org"
client.username_pw_set('client', '1234')
client.connect(rabitmq_ip, 1883, 60)
print("Connected")

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
# client.loop_forever()
client.loop_start()

user_auth = {'username': 'client', 'password': '1234'}

publish.single("demo.key", "payload", auth=user_auth)
time.sleep(50)

client.loop_stop()
