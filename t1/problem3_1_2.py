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
        self.split_area()

    def run(self):
        # for epi in range(self.total_iter_num):
        for area in range(self.area_num):
            self.area_ts(area)
        # print("第{}次迭代：".format(epi))
        print("线的斜率, k1: {}, k2: {}".format(self.k1, self.k2))
        print("每个区域的路程: {}, {}, {}, {}, 总路程: {}".format(self.area_distance[0], self.area_distance[1],
                                                        self.area_distance[2], self.area_distance[3],
                                                        self.area_distance[1] + self.area_distance[2] +
                                                        self.area_distance[0] + self.area_distance[3]))
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
        print("n: {}".format(min(self.area_distance) / max(self.area_distance)))
        print()
        if min(self.area_distance) / max(self.area_distance) >= self.eta:
            pass
        else:
            self.k1 = np.random.uniform(self.k1_range[0], self.k1_range[1])
            self.k2 = np.random.uniform(self.k2_range[0], self.k2_range[1])
            self.split_area()

    def area_ts(self, i_area):
        if self.area_node_num[i_area] == 1 or self.area_node_num[i_area] == 2:
            return
        for i in range(self.iter_num):
            # 产生can_n个新解
            can_num = 0
            can_n_array = []
            can_n_dis_array = []
            while can_num < self.can_n:
                u = np.random.randint(0, high=self.area_node_num[i_area], size=None)
                if u == self.area_node_num[i_area] - 1:
                    v = u
                    u = np.random.randint(0, high=v, size=None)
                else:
                    v = np.random.randint(u + 1, high=self.area_node_num[i_area], size=None)
                temp = []
                for j in range(u):
                    temp.append(self.area_result[i_area][j])
                j = v
                while j >= u:
                    temp.append(self.area_result[i_area][j])
                    j -= 1
                for j in range(v + 1, self.area_node_num[i_area]):
                    temp.append(self.area_result[i_area][j])
                temp = np.array(temp)
                new_dis = self.get_area_distance(temp)

                if list(temp) not in self.tabu:
                    can_n_array.append(temp)
                    can_n_dis_array.append(new_dis)
                    can_num += 1
                # else:
                #     print("in")
            temp_index = 0
            now_best_dis = can_n_dis_array[temp_index]
            for j in range(len(can_n_dis_array)):
                if can_n_dis_array[j] < now_best_dis:
                    now_best_dis = can_n_dis_array[j]
                    temp_index = j
            now_best_array = can_n_array[temp_index]
            # TODO 比较两个self.area_result 和当前结果的好坏
            if now_best_dis < self.area_distance[i_area]:
                self.area_result[i_area] = now_best_array
                self.area_distance[i_area] = now_best_dis
            for j in range(can_num):
                self.tabu_index = self.tabu_index % self.tabu_len
                self.tabu[self.tabu_index] = now_best_array.tolist()
                self.tabu_index += 1
            # print("area: {}, iter: {}, distance: {},  ".format(i_area, i, self.area_distance[i_area]))

    def split_area(self):
        area1 = [5, 10, 12, 13, 16, 27]
        area2 = [3, 4, 21, 22, 23, 24, 28]
        area3 = [17, 18, 19, 20, 25, 26, 29]
        area4 = [1, 2, 6, 7, 8, 9, 11, 14, 15]
        # for i in range(0, self.node_num - 1):
        #     if self.node_loc[i][1] >= self.get_k1_y(self.node_loc[i][0]) and self.node_loc[i][1] > self.get_k2_y(
        #             self.node_loc[i][0]):
        #         area1.append(i)
        #     elif self.get_k1_y(self.node_loc[i][0]) > self.node_loc[i][1] >= self.get_k2_y(self.node_loc[i][0]):
        #         area2.append(i)
        #     elif self.node_loc[i][1] <= self.get_k1_y(self.node_loc[i][0]) and self.node_loc[i][1] < self.get_k2_y(
        #             self.node_loc[i][0]):
        #         area3.append(i)
        #     else:
        #         area4.append(i)

        self.area_nodes[0] = area1
        self.area_nodes[1] = area2
        self.area_nodes[2] = area3
        self.area_nodes[3] = area4
        self.area_result[0] = area1
        self.area_result[1] = area2
        self.area_result[2] = area3
        self.area_result[3] = area4
        self.area_node_num[0] = len(area1)
        self.area_node_num[1] = len(area2)
        self.area_node_num[2] = len(area3)
        self.area_node_num[3] = len(area4)
        self.area_distance[0] = self.get_area_distance(area1)
        self.area_distance[1] = self.get_area_distance(area2)
        self.area_distance[2] = self.get_area_distance(area3)
        self.area_distance[3] = self.get_area_distance(area4)

    def get_k1_y(self, x):
        return self.k1 * x + self.center_loc[1] - self.center_loc[0] * self.k1

    def get_k2_y(self, x):
        return self.k2 * x + self.center_loc[1] - self.center_loc[0] * self.k2

    # def get_area_distance(self, node_array):
    #     dis = 0
    #     dis += self.get_distance(self.center_loc[0], self.center_loc[1], node_array[0][0], node_array[0][1])
    #     for i in range(len(node_array)):
    #         if i == len(node_array) - 1:
    #             loc1 = node_array[i]
    #             loc2 = self.center_loc
    #         else:
    #             loc1 = node_array[i]
    #             loc2 = node_array[i + 1]
    #
    #         dis += self.get_distance(loc1[0], loc1[1], loc2[0], loc2[1])
    #     return dis

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
    mts.run()
