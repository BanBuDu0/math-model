import pandas as pd
import numpy as np
import math
import time

path = "data/B题附件1.xlsx"
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

"""
N:城市数
s:二进制表示，遍历过得城市对应位为1，未遍历为0
dp:动态规划的距离数组
dist：城市间距离矩阵
sumpath:目前的最小路径总长度
Dtemp：当前最小距离
path:记录下一个应该到达的城市
"""

N = train_v.shape[0]
path = np.ones((2 ** (N + 1), N))
dp = np.ones((2 ** (train_v.shape[0] + 1), train_d.shape[0])) * -1


def TSP(s, init, num):
    if dp[s][init] != -1:
        return dp[s][init]
    if s == (1 << (N)):
        return dist[0][init]
    sumpath = 1000000000
    for i in range(N):
        if s & (1 << i):
            m = TSP(s & (~(1 << i)), i, num + 1) + dist[i][init]
            if m < sumpath:
                sumpath = m
                path[s][init] = i
    dp[s][init] = sumpath
    return dp[s][init]


if __name__ == "__main__":
    init_point = 0
    s = 0
    for i in range(1, N + 1):
        s = s | (1 << i)
    start = time.clock()
    distance = TSP(s, init_point, 0)
    end = time.clock()
    s = 0b11111111110
    init = 0
    num = 0
    print(distance)
    while True:
        print(path[s][init])
        init = int(path[s][init])
        s = s & (~(1 << init))
        num += 1
        if num > 9:
            break
    print("程序的运行时间是：%s" % (end - start))
