import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class GA:
    def __init__(self, m, g, pc, pm):
        """
        遗传算法
        :param m: 种群大小
        :param g:最大代数
        :param pc:交叉率
        :param pm:变异率
        """
        self.df = None
        self.node_num = None
        self.node_name = None
        self.node_loc = None
        self.result = None
        self.res_distance = None

    def init(self):
        path = "data/B题附件1.xlsx"
        self.df = pd.read_excel(path, sheet_name=0)
        self.node_num = self.df.shape[0]
        self.node_loc = np.array(self.df.iloc[:, 1:3])
        self.node_name = np.array(self.df.iloc[:, 0])
        # TODO 选择初始解为顺序排列
        self.result = np.arange(self.node_num)
        self.res_distance = self.get_result_distance(self.result)

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
