import pandas as pd
import numpy as np
import math
import time

'''
D:\Anaconda3\python.exe D:/jupyter_project/math/t1/greedy.py
结果：
531.232814976417
0 
10 
16 
27 
12 
8 
9 
20 
19 
1 
2 
17 
29 
5 
13 
4 
21 
3 
22 
23 
24 
28 
15 
11 
6 
7 
18 
26 
25 
14 
程序的运行时间是：0.0

Process finished with exit code 0

'''

path = "data/B题附件1.xlsx"
dataframe = pd.read_excel(path, sheet_name=0)
v = dataframe.iloc[:, 1:3]

train_v = np.array(v)
# train_v = np.array([[6734, 1453], [2233, 10], [5530, 1424], [401, 841],
#         [3082, 1644], [7608, 4458],
#         [7573, 3716], [7265, 1268],
#         [6898, 1885], [1112, 2049],
#         [5468, 2606], [5989, 2873],
#         [4706, 2674], [4612, 2035],
#         [6347, 2683], [6107, 669],
#         [7611, 5184], [7462, 3590],
#         [7732, 4723], [5900, 3561],
#         [4483, 3369], [6101, 1110],
#         [5199, 2182], [1633, 2809],
#         [4307, 2322], [675, 1006],
#         [7555, 4819], [7541, 3981],
#         [3177, 756], [7352, 4506],
#         [7545, 2801], [3245, 3305],
#         [6426, 3173], [4608, 1198],
#         [23, 2216], [7248, 3779],
#         [7762, 4595], [7392, 2244],
#         [3484, 2829], [6271, 2135],
#         [4985, 140], [1916, 1569],
#         [7280, 4899], [7509, 3239],
#         [10, 2676], [6807, 2993],
#         [5185, 3258], [3023, 1942]])
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
# def get_distance(x1, y1, x2, y2):
#     return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


# 计算距离矩阵
for i in range(train_v.shape[0]):
    for j in range(train_d.shape[0]):
        dist[i, j] = get_distance(train_v[i, 0], train_v[i, 1], train_d[j, 0], train_d[j, 1])



"""
s:已经遍历过的城市
dist：城市间距离矩阵
sumpath:目前的最小路径总长度
Dtemp：当前最小距离
flag：访问标记
"""
i = 1
n = train_v.shape[0]
j = 0
sumpath = 0
s = []
s.append(0)
start = time.time()
while True:
    k = 1
    Detemp = 10000000
    while True:
        l = 0
        flag = 0
        if k in s:
            flag = 1
        if (flag == 0) and (dist[k][s[i - 1]] < Detemp):
            j = k;
            Detemp = dist[k][s[i - 1]];
        k += 1
        if k >= n:
            break;
    s.append(j)
    i += 1;
    sumpath += Detemp
    if i >= n:
        break;
sumpath += dist[0][j]
end = time.time()
print("结果：")
print(sumpath)
for m in range(n):
    print("%s " % (s[m]))
print("程序的运行时间是：%s" % (end - start))
