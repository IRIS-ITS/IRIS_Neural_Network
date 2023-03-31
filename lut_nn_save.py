import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import matplotlib.pyplot as plt
import csv
import pandas as pd


sudut = None
jarak_pixel = None

theta_max = 360
r_maks = 3200


data = pd.read_csv('test.csv')
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

# for epoch in range(300000):
#     optimizer.zero_grad()
#     pred = model(data_x)
#     loss = mse_loss(pred, data_y)
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 1000 == 0:
#         print("Epoch %d  Loss %f" % (epoch + 1, loss.item()))


# torch.save(model, 'data_dimas.pth')

model = torch.load('data_dimas.pth')
model.eval()

lut = np.empty(r_maks*theta_max, dtype=np.int16)
input_buffer = np.random.random_sample((2, 2))

th_iter = 0
while th_iter < theta_max:
    r_iter = 500
    while r_iter < r_maks:
        index = th_iter * r_maks + r_iter
        input_buffer[0, 0] = (th_iter/180) - 1
        input_buffer[0, 1] = (r_iter/10/160) - 1
        input_buffer = torch.from_numpy(
            np.array(input_buffer, dtype=np.float32))

        pred = model(input_buffer[0, :])
        r_lap = pred.data * 600 + 600

        print(th_iter, " ", r_iter / 10, " ", r_lap, " ", index)

        lut[index] = r_lap
        r_iter += 1
    th_iter += 1


# with open('./LUT_fr2lap.csv', 'w', encoding='UTF8') as f:
#     writer = csv.writer(f)
#     writer.writerow(lut)

lut.tofile("aww.bin")

th_ex = 260
r_fr_ex = 116.5

r_fr_ex *= 10

print(lut[th_ex * r_maks + r_fr_ex])

exit()


############################################


x_max = 200
x_min = -200
y_max = 200
y_min = -200

frame = 170
x_lap = []
y_lap = []
x_frame = []
y_frame = []
x_lap_lama = []
y_lap_lama = []
i = 0
while i <= 360:
    input_buffer = np.random.random_sample((2, 2))

    input_buffer[0, 0] = (i/180) - 1
    input_buffer[0, 1] = (frame/160) - 1
    input_buffer = torch.from_numpy(np.array(input_buffer, dtype=np.float32))
    out_temp = np.random.random_sample((2, 1))

    out_temp = torch.from_numpy(np.array(out_temp, dtype=np.float32))
    pred = model(input_buffer[0, :])
    lapangan = pred.data * 600 + 600

    # lapangan
    x = math.cos(i * math.pi / 180) * lapangan
    y = math.sin(i * math.pi / 180) * lapangan
    x_lap.append(x)
    y_lap.append(y)

    ################################################################

    input_buffer = np.random.random_sample((2, 2))

    input_buffer[0, 0] = (i/180) - 1
    input_buffer[0, 1] = (frame/160) - 1
    input_buffer = torch.from_numpy(np.array(input_buffer, dtype=np.float32))
    out_temp = np.random.random_sample((2, 1))

    out_temp = torch.from_numpy(np.array(out_temp, dtype=np.float32))
    pred = model_lama(input_buffer[0, :])
    lapangan = pred.data * 600 + 600

    # lapangan
    x = math.cos(i * math.pi / 180) * lapangan
    y = math.sin(i * math.pi / 180) * lapangan
    x_lap_lama.append(x)
    y_lap_lama.append(y)

    # frame
    x = math.cos(i * math.pi / 180) * frame
    y = math.sin(i * math.pi / 180) * frame
    x_frame.append(x)
    y_frame.append(y)
    i += 1

x_axis_x = [x_max, x_min]
x_axis_y = [0, 0]

y_axis_x = [0, 0]
y_axis_y = [y_max, y_min]


plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.plot(x_lap, y_lap, color='g', label='lapangan')
# plt.plot(x_lap_lama, y_lap_lama, color='b', label='lapangan_lama')
plt.plot(x_frame, y_frame, color='r', label='frame')
plt.plot(x_axis_x, x_axis_y, color='b', label='sb x')
plt.plot(y_axis_x, y_axis_y, color='b', label='sb y')
plt.show()
