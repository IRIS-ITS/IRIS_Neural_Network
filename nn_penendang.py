import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd


power_test = None
jarak_test = None

power_max = 50
jarak_max = 800


data = pd.read_csv('102kicker.csv')
data_x = np.array([data.values[:, 0], data.values[:, 1]], dtype=np.float32)
data_y = np.array([data.values[:, 2]], dtype=np.float32)
data_x = data_x.transpose()
data_y = data_y.transpose()

data_x = torch.from_numpy(data_x)
data_y = torch.from_numpy(data_y)


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.act_func = nn.Sigmoid()
        self.layer1 = nn.Linear(2, 25)
        self.layer2 = nn.Linear(25, 25)
        self.layer3 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.act_func(self.layer1(x))
        x = self.act_func(self.layer2(x))
        x = self.layer3(x)
        return x


model = NeuralNet()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Normal 1000000

for epoch in range(33333):
    optimizer.zero_grad()
    pred = model(data_x)
    loss = mse_loss(pred, data_y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print("Epoch %d  Loss %f" % (epoch + 1, loss.item()))


torch.save(model, 'data_tendang.pth')

# UNTUK EXPORT KE LUT

lut = np.empty(power_max*jarak_max + jarak_max, dtype=int)
lut = np.zeros_like(lut)
input_buffer = np.random.random_sample((2, 2))

power_iter = 6
while power_iter <= power_max:
    jarak_iter = 30
    while jarak_iter < jarak_max:
        index = power_iter * jarak_max + jarak_iter
        input_buffer[0, 0] = (power_iter/25) - 1
        input_buffer[0, 1] = (jarak_iter/400) - 1
        input_buffer = torch.from_numpy(
            np.array(input_buffer, dtype=np.float32))

        pred = model(input_buffer[0, :])
        tinggi_tendang = pred.data * 450 + 450

        if tinggi_tendang < 100:
            tinggi_tendang = 100
        elif tinggi_tendang > 900:
            tinggi_tendang = 900

        lut[index] = tinggi_tendang
        jarak_iter += 1
    power_iter += 1

lut.tofile("kicker102.bin")

# with open('./LUT_kicker2.csv', 'w', encoding='UTF8') as f:
#     writer = csv.writer(f)
#     writer.writerow(lut)

# ==========================================================================================

# UNTUK TESTING MODEL
# while True:
#     print("Power : ")
#     power_test = int(input())
#     print("Jarak : ")
#     jarak_test = int(input())
#     input_buffer = np.random.random_sample((2, 2))
#     input_buffer[0, 0] = (power_test/25) - 1
#     input_buffer[0, 1] = (jarak_test/400) - 1
#     input_buffer = torch.from_numpy(
#         np.array(input_buffer, dtype=np.float32))

#     pred = model(input_buffer[0, :])
#     print((pred.data * 450) + 450)


# UNTUK TESTING LUT
while True:
    print("Power : ")
    power_test = int(input())
    print("Jarak : ")
    jarak_test = int(input())

    if power_test < 0:
        power_test = 0
    elif power_test > power_max:
        power_test = power_max

    if jarak_test < 0:
        jarak_test = 0
    elif jarak_test >= jarak_max:
        jarak_test = jarak_max - 1

    print(lut[power_test * jarak_max + jarak_test])

exit()
