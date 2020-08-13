import requests
import pandas as pd
import numpy as np
import json


class Solution:
    def __init__(self):
        """
        node_num: 包括起点在内的节点数
        node_name: 包括起点在内的节点名
        node_loc：包括起点的节点经纬度
        center_loc: 起点的经纬度
        distance_array: 各点之间的距离
        """
        self.node_num = None
        self.node_name = None
        self.node_loc = None
        self.center_loc = None
        self.distance_array = None
        self.load_data_from_csv()

    @staticmethod
    def get_road_from_baidu(loc1, loc2):
        """
        从百度api获取路径规划
        :param loc1: 起点经纬度[120.234488, 30.313231]
        :param loc2: 终点经纬度[120.234488, 30.313231]
        :return:
        """
        if loc1[1] == loc2[1] and loc1[0] == loc2[0]:
            return 0
        s1 = str(loc1[1]) + "," + str(loc1[0])
        s2 = str(loc2[1]) + "," + str(loc2[0])
        ak = "iz1iVNn7OzG6iBc1wA5e0naACuEbq8qD"
        url = "http://api.map.baidu.com/direction/v2/driving?tactics=2&origin={}&destination={}&ak={}".format(s1, s2,
                                                                                                              ak)
        result = requests.get(url)
        result_json = json.loads(result.text)
        s = result_json['result']['routes'][0]['distance']
        return s

    def load_data_from_csv(self):
        """
        从本地获取各个地点的经纬度
        :return:
        """
        path = "data/data.xlsx"
        df = pd.read_excel(path, sheet_name=0)
        self.node_num = df.shape[0]
        self.node_loc = np.array(df.iloc[:, 1:3])
        self.node_name = np.array(df.iloc[:, 0])
        self.center_loc = np.squeeze(np.array(df.iloc[0:1, 1:3]))
        self.distance_array = np.zeros((len(self.node_loc), len(self.node_loc)))

    def test(self):
        for i in range(len(self.node_loc)):
            for j in range(len(self.node_loc)):
                self.distance_array[i][j] = self.get_road_from_baidu(self.node_loc[i], self.node_loc[j])
        print(self.distance_array)


if __name__ == '__main__':
    solution = Solution()
    solution.test()
    a = [[0, 10571, 2647, 1222, 5903, 6779, 11864, 25336, 11274, 6510, 17803, 22159],
         [10408, 0, 9488, 9616, 5241, 8892, 17057, 18922, 17089, 15971, 23150, 13833],
         [2061, 9569, 0, 2141, 4901, 7453, 13925, 24118, 10351, 8571, 16880, 21157],
         [1767, 9982, 3316, 0, 5314, 6190, 12039, 24954, 11985, 7050, 18514, 21570],
         [5776, 5318, 4856, 4984, 0, 6144, 13647, 22643, 13638, 11339, 19699, 16902],
         [7852, 9442, 8673, 7060, 7087, 0, 11207, 26767, 17659, 10903, 24188, 18619],
         [12554, 17071, 14530, 12169, 13926, 10550, 0, 34396, 23039, 8505, 28770, 28399],
         [24708, 19077, 23690, 24297, 23165, 26749, 34938, 0, 29008, 30845, 28915, 22328],
         [11015, 16998, 10106, 11033, 13502, 16424, 22688, 29126, 0, 15461, 7123, 28782],
         [7170, 16344, 9146, 7721, 11676, 10286, 8781, 31782, 15808, 0, 20370, 27932],
         [17575, 22734, 15791, 17713, 19238, 22984, 29094, 28848, 7202, 20873, 0, 33558],
         [21252, 13185, 20332, 20460, 16085, 18853, 27587, 21761, 28060, 26815, 33024, 0]]
