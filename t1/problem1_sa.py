import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time


class SA:
    def __init__(self):
        self.node_num = None
        self.node_name = None
        self.node_loc = None
        self.result = None
        self.res_distance = None
        self.iter_num = 20000
        self.T = 1
        self.alpha = 0.99
        self.e = 1.0000e-30

    def init(self):
        path = "data/B题附件1.xlsx"
        df = pd.read_excel(path, sheet_name=0)
        self.node_num = df.shape[0]
        self.node_loc = np.array(df.iloc[:, 1:3])
        self.node_name = np.array(df.iloc[:, 0])
        # TODO 选择初始解为顺序排列
        self.result = np.arange(self.node_num)
        self.res_distance = self.get_result_distance(self.result)

    def run(self):
        start = time.time()
        for i in range(self.iter_num):
            # 产生新解
            u = np.random.randint(1, high=self.node_num, size=None)
            if u == self.node_num - 1:
                v = u
                u = np.random.randint(1, high=v, size=None)
            else:
                v = np.random.randint(u + 1, high=self.node_num, size=None)
            temp = []
            for j in range(u):
                temp.append(self.result[j])
            j = v
            while j >= u:
                temp.append(self.result[j])
                j -= 1
            for j in range(v + 1, self.node_num):
                temp.append(self.result[j])
            temp = np.array(temp)
            new_dis = self.get_result_distance(temp)
            df = new_dis - self.res_distance
            if df < 0:
                self.res_distance = new_dis
                self.result = temp
            elif np.exp(-df / self.T) > np.random.random():
                self.res_distance = new_dis
                self.result = temp
            now = time.time()
            if i % 100 == 0:
                print("第" + str(i) + "次迭代：")
                print("路径: {}".format(self.result))
                print("距离: {}".format(self.res_distance))
                print("运行时间：{}".format(now - start))
                print()
            self.T = self.alpha * self.T
            if self.T < self.e:
                break

    def draw(self):
        x = []
        y = []
        x0 = self.node_loc[0][0]
        y0 = self.node_loc[0][1]
        for i in self.result:
            x.append(self.node_loc[i][0])
            y.append(self.node_loc[i][1])
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.plot(x0, y0, '^')

        line1 = [(x0, y0), (x[0], y[0])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))
        for i in range(len(x) - 1):
            line1 = [(x[i], y[i]), (x[i + 1], y[i + 1])]
            (line1_xs, line1_ys) = zip(*line1)
            ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))
        line1 = [(x[len(x) - 1], y[len(x) - 1]), (x0, y0)]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))
        plt.show()

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

    def test(self, array):
        dis = 0
        for i in range(len(array)):
            if i == len(array) - 1:
                node1 = array[i]
                node2 = 0
                loc1 = self.node_loc[node1]
                loc2 = self.node_loc[node2]
            else:
                node1 = array[i]
                node2 = array[i + 1]
                loc1 = self.node_loc[node1]
                loc2 = self.node_loc[node2]
            dis += self.get_distance(loc1[0], loc1[1], loc2[0], loc2[1])
        return dis


if __name__ == '__main__':
    sa = SA()
    sa.init()
    sa.run()
    sa.draw()
