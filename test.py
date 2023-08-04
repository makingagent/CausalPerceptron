import random
import numpy as np
from perceptron import Perceptron


'''
    Test 1
    Verify that when the output is a factor of the input, perceptron can learn x->x
    otherwise output is constant
'''
p0 = Perceptron([])
p1 = Perceptron([p0])

cnt = 0
while True:
    p0.action = p1.action
    # p0.action = random.random()

    p1.update()
    p1.cal()

    cnt += 1
    if cnt % 10000 == 0:
        for i in range(10):
            p0.action = i / 10.0
            p1.update()
            p1.cal(True)
            if i == 0 or i == 9:
                print(f'round:{cnt/10000} {i}: {p1.action}')


'''
    Test 2
    Verify that when neurons are connected in series, they can learn correctly
'''
p0 = Perceptron([])
p1 = Perceptron([p0])
p2 = Perceptron([p1])
p3 = Perceptron([p2])
# p4 = Perceptron([p3])

cnt = 0
while True:
    # print("000000000000000000")
    if random.random() < 0.9:
        p0.action = p3.action
    else:
        p0.action = 200 * random.random()

    p1.update()
    p1.cal()
    p2.update()
    p2.cal()
    p3.update()
    p3.cal()
    # p4.update()
    # p4.cal()

    cnt += 1
    if cnt % 10000 == 0:
        for i in range(10):
            p0.action = i / 10.0
            p1.update()
            p1.cal(True)
            p2.update()
            p2.cal(True)
            p3.update()
            p3.cal(True)
            # p4.update()
            # p4.cal(True)
            if i == 0 or i == 9:
                # print(f'round:{cnt/10000} {i}: {p1.action}')
                # print(f'round:{cnt/10000} {i}: {p2.action}')
                print(f'round:{cnt/10000} {i}: {p3.action}')
                # print(f'round:{cnt/10000} {i}: {p4.action}')


'''
    Test 3
    Verify that when a neuron has multiple inputs, it can learn correctly
'''
p0 = Perceptron([])
p1 = Perceptron([])
p2 = Perceptron([])
p3 = Perceptron([p0, p1, p2])

cnt = 0
while True:
    p0.action = 2 * random.random() - 1
    p1.action = p3.action
    p2.action = 2 * random.random() - 1
    p3.update()
    p3.cal()

    # print('p2: {:.2f} {:.2f}'.format(p2.weight[0], p2.bias), p2.action)

    cnt += 1
    if cnt % 10000 == 0:
        print(f'{p3.theta[0]:.3f} {p3.theta[1]:.3f} {p3.theta[2]:.3f}')


'''
    Test 4
    Verify that the neuron has the ability to avoid non-causal environments
'''
p0 = Perceptron([])
p1 = Perceptron([])
p2 = Perceptron([p0, p1])

cnt = 0
num_cnt = [0, 0, 0, 0, 0, 0, 0, 0]
total = 0
mod = 8
while True:
    if cnt % mod == 3 or cnt % mod == 4:
        p0.action = p2.action
    else:
        p0.action = 0
    
    if (cnt+1) % mod == 3 or (cnt+1) % mod == 4:
        p1.action = p2.action
    else:
        p1.action = 0
    p2.update()
    p2.cal()

    if p2.action > 0.5:
        cnt += 1
    else:
        cnt -= 1

    cnt = (cnt + mod) % mod

    num_cnt[cnt] += 1
    total += 1
    # print(f'{cnt} {p2.theta[0]} {p2.theta[1]} {p2.theta[2]} {p2.action}')
    print([f'{(num_cnt[i] / total):.3f}' for i in range(mod)])
