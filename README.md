# Robotik
Raspberry Pi Robot

**Pololu help file:**

https://www.pololu.com/docs/0J40

**OpenCV:**

pip install opencv-contrib-python

pip install opencv-python

**Idea for line finding**
https://github.com/naokishibuya/car-finding-lane-lines

**Image downloading tool**
https://github.com/hardikvasa/google-images-download

**Labeling tool**
https://github.com/tzutalin/labelimg

**Install uv4l on Raspberry**
https://www.linux-projects.org/uv4l/installation/

**Start video stream from Raspberry Pi**
uv4l --driver raspicam --auto-video_nr --encoding mjpeg –-framerate 15 --width 640 --height 480 --enable-server on

**On Client**
cap = cv2.VideoCapture('http://192.168.2.200:8080/stream/video.mjpeg')
