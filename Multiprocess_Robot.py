from Distance import Distance
from multiprocessing import Value, Process
import time

d = Distance()
x = Value('f', 0)

p = Process(target=d.measure, args=x)

for i in range(100):
    if not p.is_alive():
        p.start()

    print(x.Value)
    time.sleep(1)
