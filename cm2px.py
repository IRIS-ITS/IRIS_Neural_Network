import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd


sudut_lap = None
jarak_lap = None

theta_max = 360
r_lap_max = 1200


data = pd.read_csv('102cm2px.csv')
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
        self.layer1 = nn.Linear(2, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.act_func(self.layer1(x))
        x = self.act_func(self.layer2(x))
        x = self.layer3(x)
        return x


model = NeuralNet()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Normal 1000000
# Normal2 70000

for epoch in range(300000):
    optimizer.zero_grad()
    pred = model(data_x)
    loss = mse_loss(pred, data_y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print("Epoch %d  Loss %f" % (epoch + 1, loss.item()))


# torch.save(model, 'data_baru_inv.pth')

# UNTUK EXPORT KE LUT

lut = np.empty(r_lap_max*theta_max, dtype=np.int16)
input_buffer = np.random.random_sample((2, 2))

th_iter = 0
while th_iter < theta_max:
    r_lap_iter = 0
    while r_lap_iter < r_lap_max:
        index = th_iter * r_lap_max + r_lap_iter
        input_buffer[0, 0] = (th_iter/180) - 1
        input_buffer[0, 1] = (r_lap_iter/600) - 1
        input_buffer = torch.from_numpy(
            np.array(input_buffer, dtype=np.float32))

        pred = model(input_buffer[0, :])
        r_frame = (pred.data * 160 + 160) * 10

        lut[index] = r_frame
        r_lap_iter += 1
    th_iter += 1

# with open('./LUT_lap2fr.csv', 'w', encoding='UTF8') as f:
#     writer = csv.writer(f)
#     writer.writerow(lut)
lut.tofile("abc.bin")

# ==========================================================================================

# UNTUK TESTING MODEL
# while True:
#     print("Sudut : ")
#     sudut_lap = int(input())
#     print("Jarak Lapangan : ")
#     jarak_lap = int(input())
#     input_buffer = np.random.random_sample((2, 2))
#     input_buffer[0, 0] = (sudut_lap/180) - 1
#     input_buffer[0, 1] = (jarak_lap/600) - 1
#     input_buffer = torch.from_numpy(
#         np.array(input_buffer, dtype=np.float32))

#     pred = model(input_buffer[0, :])
#     print((pred.data * 160) + 160)


# UNTUK TESTING LUT
# while True:
#     print("Sudut : ")
#     sudut_lap = int(input())
#     print("Jarak Lapangan : ")
#     jarak_lap = int(input())

#     print(lut[sudut_lap * r_lap_max + jarak_lap])

exit()
