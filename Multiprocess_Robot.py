from Distance import Distance
from robot import Robot
import multiprocessing
import time

d = Distance()
robotik = Robot()

x = multiprocessing.Value('d', 100.0)

p = multiprocessing.Process(target=d.measure, args=(x, ))
p.start()

try:
    for i in range(10):

        if not p.is_alive():
            p = multiprocessing.Process(target=d.measure, args=(x, ))
            p.start()

        print(x.value)
        if x.value >= 50:
            robotik.go_forward()
        elif x.value < 10:
            robotik.go_backward()
        elif (x.value >= 10) and (x.value < 50):
            robotik.turn_left()

        time.sleep(0.3)

except KeyboardInterrupt:
    print('Finishing')

except:
    print('Other error')

finally:
    p.terminate()
    d.cleanup()

