from Distance import Distance
import multiprocessing
import time
import Ad

d = Distance(20, 21)

x = multiprocessing.Value('d', 100.0)

p = multiprocessing.Process(target=d.measure, args=(x, ))
p.start()

try:
    while True:

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

finally:
    p.terminate()
    robotik.stop_robot()
    d.cleanup()

