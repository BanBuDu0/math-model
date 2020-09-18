import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable


def load_data():
    s1_train_data_path = './data/S1/S1_train_data.xlsx'
    s1_train_event_path = './data/S1/S1_train_event.xlsx'

    train_x = []
    train_y = []

    for j in range(12):
        s1_train_data_df = pd.read_excel(s1_train_data_path, sheet_name=j, header=None)
        s1_train_event_df = pd.read_excel(s1_train_event_path, sheet_name=j, header=None)

        s1_train_data_np = s1_train_data_df.to_numpy()
        s1_train_event_np = s1_train_event_df.to_numpy()

        y = s1_train_event_np[0][0]
        t1 = s1_train_event_np[0][1]

        idx = s1_train_event_np[1:].reshape(5, -1, 2)

        res = []
        for i in idx:
            a = s1_train_data_df.iloc[t1 - 1: int(i[-1:, 1]), :].to_numpy()
            a = a[-510:, :]
            res.append(a)
            t1 = int(i[-1:, 1])
        res = np.array(res)
        train_x.append(res)
        train_y.append(y)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # (1*28*28)
                in_channels=5,
                out_channels=16,
                kernel_size=5,
                stride=1,  # 步长
                padding=2,
            ),  # (16*28*28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (16*14*14)
        )
        self.conv2 = nn.Sequential(  # 16*14*14
            nn.Conv2d(16, 32, 5, 1, 2),  # 32*14*14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32*7*7
        )
        self.out = nn.Linear(32 * 127 * 5, 10)  # 全连接

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch,32,7,7)
        print(x.shape)
        x = x.view(x.size(0), -1)  # (batch,32*7*7)
        print(x.shape)
        output = self.out(x)
        return output


if __name__ == '__main__':
    t_x, t_y = load_data()
    cnn = CNN()
    train_t = torch.tensor(t_x).float()
    res = cnn(train_t)
