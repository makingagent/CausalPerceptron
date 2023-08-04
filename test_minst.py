import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from display_data import display_data
from perceptron import Perceptron
from time import sleep


data = sio.loadmat('ex3data1.mat')
X = data['X']; y = data['y'] % 10

# init network
network = [[], [], []]
for i in range(400):
    network[0].append(Perceptron([]))
for i in range(25):
    network[1].append(Perceptron(network[0]))
for i in range(10):
    network[2].append(Perceptron(network[1]))

# check
def check():
    rand_indices = np.random.permutation(X.shape[0])
    test = X[rand_indices[:100]]

    total = 0
    cnt = [0 for i in range(10)]
    for i in range(100):
        max_val = np.max(np.abs(test[i]))
        test[i] = np.abs(np.round(test[i]/max_val))

        for j in range(400):
            network[0][j].action = test[i][j]
        for j in range(25):
            network[1][j].update()
            network[1][j].cal(True)
        for j in range(10):
            network[2][j].update()
            network[2][j].cal(True)
        check = True
        real = int(y[rand_indices[i]][0])

        # display input img
        # display = display_data(test[[i]])
        # plt.imshow(display, cmap='gray', vmin = 0, vmax = 1)
        # plt.axis('off')
        # plt.pause(1)
        # input()
        # print(f'real: {real}')

        for j in range(10):
            if j != real and network[2][j].action + 1e-8 > network[2][real].action:
                check = False
        if network[2][real].action < 0.5 + 1e-8:
            check = False
        if check:
            cnt[real] += 1
            total += 1

    print('check res:', total, [cnt[i] for i in range(10)])

sx = X
sy = y
select = -1
rcnt = 0

while True:

    if select != -1:
        sel = sx[[select]]
        max_val = np.max(np.abs(sel[0]))
        sel[0] = np.abs(np.round(sel[0]/max_val))
    else:
        sel = np.zeros((1,400))

    # run network
    for i in range(400):
        network[0][i].action = sel[0][i]
    for i in range(25):
        network[1][i].update()
        network[1][i].cal()
    for i in range(10):
        network[2][i].update()
        network[2][i].cal()

    # select one number in 0~9
    mx = -1
    idx = -1
    for i in range(10):
        if network[2][i].action > mx and network[2][i].action > 0.5:
            mx = network[2][i].action
            idx = i
    # print(idx)

    if rcnt % 1000 == 0:
        print(f'round: {rcnt//1000}')
        check()
        sleep(1)
    rcnt += 1

    # if selected number X, than show random img of X
    if idx != -1:
        sy = y == idx
        sx = X[sy.T[0]]
        select = random.randint(1, sx.shape[0]) - 1
    else:
        select = -1
