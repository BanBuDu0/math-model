import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

path = "./B题附件1.xlsx"
data = pd.read_excel(path, sheet_name=0)
node_num = data.shape[0]

x = data['传感器经度']
y = data['传感器纬度']

graph = np.zeros((node_num, node_num))


# def get_distance(x1, y1, x2, y2):
#     return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def get_distance(x1, y1, x2, y2):
    # x,y分别表示一个经纬度坐标点
    R = 6371
    theta = math.acos(math.sin(x1) * math.sin(x2) + (math.cos(x1) * math.cos(x2) * math.cos(y1 - y2)))
    L = theta * R
    return L


for i in range(node_num):
    for j in range(i, node_num):
        if i == j:
            graph[i][j] = 0
        else:
            graph[i][j] = get_distance(x[i], y[i], x[j], y[j])
            graph[j][i] = graph[i][j]

for i in graph:
    i[0] = np.inf

graph_out = graph[0:1][:]
graph_out = np.hstack((graph_out, np.array(np.inf).reshape(1, -1)))
graph = np.row_stack((graph, np.full(node_num, np.inf)))
graph = np.hstack((graph, graph_out.reshape(-1, 1)))
# print(graph)
graph[0][0] = 0
graph[30][30] = 0
visited = np.zeros(node_num + 1)


def floyd(g):
    a = [[0 for i in range(31)] for i in range(31)]
    # a = np.zeros((node_num + 1, node_num + 1))
    p = [[0 for i in range(31)] for i in range(31)]
    for i in range(node_num + 1):
        for j in range(node_num + 1):
            a[i][j] = g[i][j]
            p[i][j] = -1
    for k in range(node_num + 1):
        for i in range(node_num + 1):
            for j in range(node_num + 1):
                s = a[i][j]
                s1 = a[i][k]
                s2 = a[k][j]
                if a[i][j] > a[i][k] + a[k][j]:
                    a[i][j] = a[i][k] + a[k][j]
                    p[i][j] = k
    return a, p


if __name__ == '__main__':
    res_g, res_p = floyd(graph)
    t = np.array(res_g)
    t1 = np.array(res_p)
    print(t)
    print(res_p)
