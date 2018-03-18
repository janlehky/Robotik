from Distance import Distance
import multiprocessing
import time

d = Distance()


# def f(data, name):
#     time.sleep(2)
#     data.value += 1


x = multiprocessing.Value('d', 0.0)

p = multiprocessing.Process(target=d.measure(), args=(x, 'x'))
p.start()

for i in range(100):

    if not p.is_alive():
        p = multiprocessing.Process(target=d.measure(), args=(x, 'x'))
        p.start()

    print(x.value)
    time.sleep(1)
