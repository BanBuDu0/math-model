import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


class solution:
    def __init__(self, f, v, r):
        self.node_num = None
        self.node_name = None
        self.node_loc = None
        self.result = None
        self.res_distance = None
        self.node_consume = None
        self.a = None
        self.total_dis = 407.48987719505214
        self.f = f
        self.v = v
        self.r = r
        self.max_battery_capacity = None
        # total_consume记录当前时间点每个node的总消耗
        self.total_consume = None

    #
    # def cycle_do(self):
    #     ii = 0
    #     T2 = np.zeros(30)
    #     for _ in range(20):
    #         for i in range(self.node_num):
    #             node_total = 0
    #             # from i to i+1,and charge for i+1
    #             if i == self.node_num - 1:
    #                 node1 = self.node_loc[i]
    #                 node2 = self.node_loc[0]
    #             else:
    #                 node1 = self.node_loc[i]
    #                 node2 = self.node_loc[i + 1]
    #                 pass
    #             dis = self.total_dis
    #             # t1 在路上消耗的时间
    #             t1 = dis / self.v
    #             # j 为充电节点
    #             if i == self.node_num - 1:
    #                 j = 0
    #             else:
    #                 j = i + 1
    #             # t2 为节点j充电的时间
    #             if j == 0:
    #                 t2 = 0
    #                 T2[j] = 0
    #             else:
    #                 t2 = (self.node_consume[j] * t1 + self.node_consume[j] * (np.sum(T2) - T2[j])) / (
    #                         self.r - self.node_consume[j])
    #                 T2[j] = t2
    #             node_total = self.node_consume[j] * t1 + self.node_consume[j] * (np.sum(T2) - T2[j]) + self.f
    #             if self.max_battery_capacity[j] < node_total:
    #                 self.max_battery_capacity[j] = node_total
    #         print(self.max_battery_capacity)
    #     return (t1, T2, self.max_battery_capacity)

    def try_do(self):
        self.a = [[0, 5, 13, 16, 27, 12, 10], [0, 21, 4, 23, 24, 28, 22, 3], [0, 17, 20, 19, 18, 25, 26, 29],
                  [0, 1, 9, 7, 6, 14, 11, 15, 8, 2]]
        for j in range(4):
            self.road = self.a[j]
            temp = []
            for i in self.node_consume:
                temp.append(i)
            for i in range(len(self.node_consume)):
                if i not in self.road:
                    self.node_consume[i] = 0
            for _ in range(50):
                for p in range(len(self.road)):
                    # 从 i -> i + 1, 并给 i + 1 充电
                    now_visit = self.road[p]
                    next_visit = self.road[((p + 1) % len(self.road))]
                    node1 = self.node_loc[now_visit]
                    node2 = self.node_loc[next_visit]
                    dis = self.get_distance(node1[0], node1[1], node2[0], node2[1]) * 1000
                    # t1 = 在路上消耗的时间
                    t1 = dis / self.v
                    timestamp_walk_consume = self.node_consume * t1
                    self.total_consume += timestamp_walk_consume
                    # j = 充电的节点

                    if self.max_battery_capacity[next_visit] < self.total_consume[next_visit]:
                        self.max_battery_capacity[next_visit] = self.total_consume[next_visit]
                    # t2 = 给 j 充电的时间
                    t2 = self.total_consume[next_visit] / self.r
                    timestamp_charge_consume = self.node_consume * t2
                    self.total_consume += timestamp_charge_consume
                    self.total_consume[next_visit] = 0
            for i in range(len(temp)):
                self.node_consume[i] = temp[i]
        print(self.max_battery_capacity + self.f)
        return self.max_battery_capacity[1] + self.f

    def get_data(self):
        path1 = "data/B题附件1.xlsx"
        path2 = "data/B题附件2.xlsx"
        df = pd.read_excel(path1, sheet_name=0)
        self.node_num = df.shape[0]
        self.node_loc = np.array(df.iloc[:, 1:3])
        self.node_name = np.array(df.iloc[:, 0])
        df = pd.read_excel(path2, sheet_name=0)
        temp = np.array(df.iloc[:, 3])
        temp[0] = 0
        self.node_consume = temp.astype(np.float64)
        self.node_consume /= 60 * 60
        self.max_battery_capacity = np.zeros(self.node_num)
        self.total_consume = np.zeros(self.node_num)

    def get_result_distance(self, node_array):
        dis = 0
        for i in range(len(node_array)):
            if i == len(node_array) - 1:
                node1 = node_array[i]
                node2 = 0
                loc1 = self.node_loc[node1]
                loc2 = self.node_loc[node2]
            else:
                node1 = node_array[i]
                node2 = node_array[i + 1]
                loc1 = self.node_loc[node1]
                loc2 = self.node_loc[node2]
            dis += self.get_distance(loc1[0], loc1[1], loc2[0], loc2[1])
        return dis

    @staticmethod
    def get_distance(x1, y1, x2, y2):
        # x,y分别表示一个经纬度坐标点
        if x1 == x2 and y1 == y2:
            return 0
        R = 6371
        theta = math.acos(math.sin(x1) * math.sin(x2) + (math.cos(x1) * math.cos(x2) * math.cos(y1 - y2)))
        L = theta * R
        return L


if __name__ == '__main__':
    x = []
    y = []
    t = np.linspace(0.2, 10, 60)
    for i in t:
        s = solution(f=40, v=4, r=i)
        s.get_data()
        cap = s.try_do()
        x.append(i)
        y.append(cap)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x, y)
    plt.title("传感器节点1最小电池容量随充电速率变化情况")
    plt.xlabel("充电速率(mA/s)")
    plt.ylabel("最小电池容量(mA)")
    plt.show()

    plt.clf()
    x = []
    y = []
    t = np.linspace(5, 35, 100)
    for i in t:
        s = solution(f=40, v=i, r=0.2)
        s.get_data()
        cap = s.try_do()
        x.append(i)
        y.append(cap)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.plot(x, y)
    plt.title("传感器节点1最小电池容量随移动充电器的移动速度变化情况")
    plt.xlabel("移动速度(m/s)")
    plt.ylabel("最小电池容量(mA)")
    plt.show()
