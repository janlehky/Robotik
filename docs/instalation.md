**OpenCV:**

pip install opencv-contrib-python

pip install opencv-python

#PC

We need to have Tensorflow installed 

**On Client**
cap = cv2.VideoCapture('http://192.168.2.200:8080/stream/video.mjpeg')

**MQTT**

Install Erlang and RabitMQ based on information at http://www.rabbitmq.com/download.html

Enable RabitMQ admin interface plugin

Enable RabitMQ mqtt plugin

#Raspberry Pi

**PCA9685 install**
sudo pip install adafruit-pca9685

**Prebuilt tensorflow**
https://github.com/lhelontra/tensorflow-on-arm/releases

**Install uv4l on Raspberry**
https://www.linux-projects.org/uv4l/installation/

**Image downloading tool**
https://github.com/hardikvasa/google-images-download
googleimagesdownload --keywords "tennis ball in grass" --limit 300 --size medium --chromedriver "D:\Projects\Robot\pictures\chromedriver.exe"


**Start video stream from Raspberry Pi**
uv4l --driver raspicam --auto-video_nr --encoding mjpeg –-framerate 15 --width 1920 --height 1080 --enable-server on


**Pololu help file:**

For early version of control
https://www.pololu.com/docs/0J40