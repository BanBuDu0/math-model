import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class MTS:
    # 多区域禁忌搜索算法
    def __init__(self, area_num=4):
        self.node_num = None
        self.node_name = None
        self.node_loc = None
        self.center_loc = None
        self.area_num = area_num
        self.area_node_num = [i for i in range(self.area_num)]  # 每个区域的节点数量
        self.area_nodes = [i for i in range(self.area_num)]  # 每个区域包含的节点
        self.area_result = [i for i in range(self.area_num)]  # 每个区域的路线结果
        self.area_distance = np.zeros(area_num)  # 每个区域的距离结果
        self.k1_range = [0, 3]
        self.k2_range = [-3, 0]
        self.eta = 0.95
        self.k1 = np.random.uniform(self.k1_range[0], self.k1_range[1])
        self.k2 = np.random.uniform(self.k2_range[0], self.k2_range[1])
        self.iter_num = 500
        self.total_iter_num = 2000
        self.tabu_len = int(np.sqrt(406))  # sqrt(c(32,2)) = 20, c(32,2) = 406
        self.tabu = [i for i in range(self.tabu_len)]
        self.tabu_index = 0
        self.can_n = 10

    def init(self):
        path = "data/B题附件1.xlsx"
        df = pd.read_excel(path, sheet_name=0)
        self.node_num = df.shape[0]
        self.center_loc = np.squeeze(np.array(df.iloc[0:1, 1:3]))
        self.node_loc = np.array(df.iloc[1:, 1:3])
        self.node_name = np.array(df.iloc[:, 0])

        print("每个区域的路程: {}, {}, {}, {}, 总路程: {}".format(self.area_distance[0], self.area_distance[1],
                                                        self.area_distance[2], self.area_distance[3],
                                                        self.area_distance[1] + self.area_distance[2] +
                                                        self.area_distance[0] + self.area_distance[3]))

        self.area_result[0] = [9, 15, 26, 12]
        self.area_result[1] = [20, 22, 23, 27, 21, 2, 3, 4]
        self.area_result[2] = [28]
        self.area_result[3] = [1, 11, 7, 14, 8, 6, 10, 5, 13, 24, 17, 25, 18, 19, 0, 16]

        print("每个区域的节点:")
        fig, ax = plt.subplots()
        a1x = []
        a1y = []
        a2x = []
        a2y = []
        a3x = []
        a3y = []
        a4x = []
        a4y = []
        for i in self.area_result[0]:
            a1x.append(self.node_loc[i][0])
            a1y.append(self.node_loc[i][1])
        for i in self.area_result[1]:
            a2x.append(self.node_loc[i][0])
            a2y.append(self.node_loc[i][1])
        for i in self.area_result[2]:
            a3x.append(self.node_loc[i][0])
            a3y.append(self.node_loc[i][1])
        for i in self.area_result[3]:
            a4x.append(self.node_loc[i][0])
            a4y.append(self.node_loc[i][1])

        for p in range(self.area_num):
            print(self.area_result[p])

        ax.plot(a1x, a1y, 'o')
        ax.plot(a2x, a2y, 'p')
        ax.plot(a3x, a3y, 'v')
        ax.plot(a4x, a4y, '^')
        plt.show()
        plt.clf()

        fig, ax = plt.subplots()
        ax.plot(a1x, a1y, 'o')
        line1 = [(self.center_loc[0], self.center_loc[1]), (a1x[0], a1y[0])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))
        for i in range(len(a1x) - 1):
            line1 = [(a1x[i], a1y[i]), (a1x[i + 1], a1y[i + 1])]
            (line1_xs, line1_ys) = zip(*line1)
            ax.add_line(Line2D(line1_xs, line1_ys))
        line1 = [(a1x[len(a1x) - 1], a1y[len(a1y) - 1]), (self.center_loc[0], self.center_loc[1])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))

        ax.plot(a2x, a2y, 'p')
        line1 = [(self.center_loc[0], self.center_loc[1]), (a2x[0], a2y[0])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))
        for i in range(len(a2x) - 1):
            line1 = [(a2x[i], a2y[i]), (a2x[i + 1], a2y[i + 1])]
            (line1_xs, line1_ys) = zip(*line1)
            ax.add_line(Line2D(line1_xs, line1_ys))
        line1 = [(a2x[len(a2x) - 1], a2y[len(a2y) - 1]), (self.center_loc[0], self.center_loc[1])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))

        ax.plot(a3x, a3y, 'v')
        line1 = [(self.center_loc[0], self.center_loc[1]), (a3x[0], a3y[0])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))
        for i in range(len(a3x) - 1):
            line1 = [(a3x[i], a3y[i]), (a3x[i + 1], a3y[i + 1])]
            (line1_xs, line1_ys) = zip(*line1)
            ax.add_line(Line2D(line1_xs, line1_ys))
        line1 = [(a3x[len(a3x) - 1], a3y[len(a3y) - 1]), (self.center_loc[0], self.center_loc[1])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))

        ax.plot(a4x, a4y, '^')
        line1 = [(self.center_loc[0], self.center_loc[1]), (a4x[0], a4y[0])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))
        for i in range(len(a4x) - 1):
            line1 = [(a4x[i], a4y[i]), (a4x[i + 1], a4y[i + 1])]
            (line1_xs, line1_ys) = zip(*line1)
            ax.add_line(Line2D(line1_xs, line1_ys))
        line1 = [(a4x[len(a4x) - 1], a4y[len(a4y) - 1]), (self.center_loc[0], self.center_loc[1])]
        (line1_xs, line1_ys) = zip(*line1)
        ax.add_line(Line2D(line1_xs, line1_ys))

        ax.plot(self.center_loc[0], self.center_loc[1], 's')
        plt.show()

    def get_k1_y(self, x):
        return self.k1 * x + self.center_loc[1] - self.center_loc[0] * self.k1

    def get_k2_y(self, x):
        return self.k2 * x + self.center_loc[1] - self.center_loc[0] * self.k2

    def get_area_distance(self, node_array):
        dis = 0
        t = self.node_loc[node_array[0]][0]
        t1 = self.node_loc[node_array[0]][1]
        dis += self.get_distance(self.center_loc[0], self.center_loc[1], t,
                                 t1)
        for i in range(len(node_array)):
            if i == len(node_array) - 1:
                node1 = node_array[i]
                loc1 = self.node_loc[node1]
                loc2 = self.center_loc
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
    mts = MTS()
    mts.init()
