from tspy import TSP

import pandas as pd
import numpy as np
import math
import time

path = "./B题附件1.xlsx"
dataframe = pd.read_excel(path, sheet_name=0)
v = dataframe.iloc[:, 1:3]

train_v = np.array(v)
train_d = train_v
dist = np.zeros((train_v.shape[0], train_d.shape[0]))


def get_distance(x1, y1, x2, y2):
    # x,y分别表示一个经纬度坐标点
    if x1 == x2 and y1 == y2:
        return 0
    R = 6371
    theta = math.acos(math.sin(x1) * math.sin(x2) + (math.cos(x1) * math.cos(x2) * math.cos(y1 - y2)))
    L = theta * R
    return L


# 计算距离矩阵
for i in range(train_v.shape[0]):
    for j in range(train_d.shape[0]):
        dist[i, j] = get_distance(train_v[i, 0], train_v[i, 1], train_d[j, 0], train_d[j, 1])

tsp = TSP()

tsp.read_mat(dist)
