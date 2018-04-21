import tensorflow as tf
import cv2 as cv
import Adafruit_PCA9685
import multiprocessing
import RPi.GPIO as GPIO     #Import GPIO library
from Distance import Distance

# Set debug bit for tuning on PC
debug = 1

if not debug:
    GPIO.setmode(GPIO.BCM)  # Set GPIO pin numbering

# define values for communication between processes
right_speed = multiprocessing.Value('d', 0.0)
left_speed = multiprocessing.Value('d', 0.0)
front_distance_sensor = multiprocessing.Value('d', 0.0)


def drive_vehicle():
    """Logic for controlling car movement"""
    # Initialise the PCA9685 using the default address (0x40).
    pwm = Adafruit_PCA9685.PCA9685()

    # set number of pins for direction of drives
    left_fwd_pin = 11
    left_bwd_pin = 12

    right_fwd_pin = 13
    right_bwd_pin = 14

    GPIO.setup(left_fwd_pin, GPIO.OUT)  # left forward pin
    GPIO.setup(left_bwd_pin, GPIO.OUT)  # left backward pin

    GPIO.setup(right_fwd_pin, GPIO.OUT)  # right forward pin
    GPIO.setup(right_bwd_pin, GPIO.OUT)  # right backward pin

    while True:
        # right drives
        if right_speed == 0 or front_distance_sensor < 5:
            GPIO.output(right_fwd_pin, False)
            GPIO.output(right_bwd_pin, False)
        else:
            if right_speed > 0:
                GPIO.output(right_fwd_pin, True)
                GPIO.output(right_bwd_pin, False)
            elif right_speed < 0:
                GPIO.output(right_fwd_pin, False)
                GPIO.output(right_bwd_pin, True)

            pwm.set_pwm(0, 0, right_speed)
            pwm.set_pwm(1, 0, right_speed)

        # left drives
        if left_speed == 0 or front_distance_sensor < 5:
            GPIO.output(left_fwd_pin, False)
            GPIO.output(left_bwd_pin, False)
        else:
            if left_speed > 0:
                GPIO.output(left_fwd_pin, True)
                GPIO.output(left_bwd_pin, False)
            elif left_speed < 0:
                GPIO.output(left_fwd_pin, False)
                GPIO.output(left_bwd_pin, True)

            pwm.set_pwm(5, 0, left_speed)
            pwm.set_pwm(6, 0, left_speed)


if not debug:
    # Create distance measurement process and start it
    drive_process = multiprocessing.Process(target=drive_vehicle, args=(right_speed, left_speed, front_distance_sensor, ))
    drive_process.start()

    # Create measurement object for front sensor
    front_measurement = Distance(20, 21)
    # Create front measurement process and start it
    front_measurement_process = multiprocessing.Process(target=front_measurement.measure, args=(front_distance_sensor, ))
    front_measurement_process.start()

# Read the graph.
with tf.gfile.FastGFile('models/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    cap = cv.VideoCapture('http://192.168.2.200:8080/stream/video.mjpeg')

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Read and preprocess an image.
            rows = frame.shape[0]
            cols = frame.shape[1]
            inp = cv.resize(frame, (300, 300))
            inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

            # Run the model
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

            # Visualize detected bounding boxes.
            num_detections = int(out[0][0])
            for i in range(num_detections):
                classId = int(out[3][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.5:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv.rectangle(frame, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=3)

                    if classId == 1:
                        middle_point_X = x + (right - x)/2
                        middle_point_Y = y + (bottom - y)/2
                        # print center point of ball
                        # print('Middle point: X:{}, Y{}'.format(middle_point_X, middle_point_Y))
                        if middle_point_X < cols/2 and middle_point_Y < rows / 2:
                            print('Upper Left')
                        elif middle_point_X >= cols/2 and middle_point_Y < rows / 2:
                            print('Upper Right')
                        elif middle_point_X < cols/2 and middle_point_Y >= rows / 2:
                            print('Lower Left')
                        elif middle_point_X >= cols / 2 and middle_point_Y >= rows / 2:
                            print('Lower Right')

            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

if not debug:
    # Terminate running processes and cleanup claimed GPIO's
    drive_process.terminate()
    front_measurement_process.terminate()
    front_measurement.cleanup()
