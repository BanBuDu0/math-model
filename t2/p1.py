import time

import requests
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import folium


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
        self.iter_num = 20000
        self.T = 1
        self.alpha = 0.999
        self.e = 1.0000e-30
        self.distance_array = None
        self.route_array = None
        self.load_data_from_csv()
        # self.test()
        self.distance_array = [[0, 10571, 2647, 1222, 5903, 6779, 11864, 25336, 11274, 6510, 17803, 22159],
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

        self.node_cap = [7, 6, 7, 9, 8, 11, 10, 8, 4, 16, 14]
        self.result, self.result_flag, self.result_capacity = self.generate_init_solution()
        self.result_distance = self.route2dis(self.result, self.result_flag, self.result_capacity)

    @staticmethod
    def get_road_from_baidu(loc1, loc2):
        """
        从百度api获取路径规划
        :param loc1: 起点经纬度[120.234488, 30.313231]
        :param loc2: 终点经纬度[120.234488, 30.313231]
        :return:
        """
        if loc1[1] == loc2[1] and loc1[0] == loc2[0]:
            return 0, []
        s1 = str(loc1[1]) + "," + str(loc1[0])
        s2 = str(loc2[1]) + "," + str(loc2[0])
        ak = "iz1iVNn7OzG6iBc1wA5e0naACuEbq8qD"
        url = "http://api.map.baidu.com/direction/v2/driving?tactics=2&origin={}&destination={}&ak={}".format(s1, s2,
                                                                                                              ak)
        result = requests.get(url)
        result_json = json.loads(result.text)
        s = result_json['result']['routes'][0]['distance']
        p = result_json['result']['routes'][0]['steps']
        ar = []
        for i in p:
            temp = str.split(i['path'], ';')
            for j in temp:
                z = j.split(',')
                ar.append([float(z[1]), float(z[0])])
        return s, ar

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
        self.route_array = [[0 for i in range(len(self.node_loc))] for i in range(len(self.node_loc))]

    def test(self):
        for i in range(len(self.node_loc)):
            for j in range(len(self.node_loc)):
                self.distance_array[i][j], self.route_array[i][j] = self.get_road_from_baidu(self.node_loc[i],
                                                                                             self.node_loc[j])
        print(self.distance_array)
        print(self.route_array)

    def generate_init_solution(self):
        all_node = []
        for i in range(self.node_num - 1):
            node = Node(self.node_cap[i], i)
            all_node.append(node)
        res = []
        res_flag = []
        res_capacity = []
        while len(all_node) != 0:
            u = np.random.randint(0, high=len(all_node), size=None)
            node = all_node[u]
            all_node.remove(all_node[u])
            if len(all_node) == 0:
                if node.cap == 5:
                    res.append(node.id)
                    res_capacity.append(5)
                    res_flag.append(False)
                else:
                    for i in range(int(node.cap / 5)):
                        res.append(node.id)
                        res_capacity.append(5)
                        res_flag.append(False)
                    res.append(node.id)
                    res_capacity.append(node.cap % 5)
                    res_flag.append(False)

                break
            if node.cap == 5:
                res.append(node.id)
                res_capacity.append(5)
                res_flag.append(False)
            elif node.cap > 5 and node.cap % 5 == 0:
                for i in range(int(node.cap / 5)):
                    res.append(node.id)
                    res_capacity.append(5)
                    res_flag.append(False)
            elif node.cap > 5:
                for i in range(int(node.cap / 5) + 1):
                    res.append(node.id)
                    res_capacity.append(5)
                    res_flag.append(False)
                remain = 5 - node.cap % 5
                while remain > 0:
                    if len(all_node) == 0:
                        print("ok1")
                        break
                    v = np.random.randint(0, high=len(all_node), size=None)
                    if all_node[v].cap > remain:
                        all_node[v].cap -= remain
                        res.append(all_node[v].id)
                        res_capacity.append(remain)
                        res_flag.append(True)
                        break
                    elif all_node[v].cap == remain:
                        all_node[v].cap -= remain
                        res.append(all_node[v].id)
                        res_flag.append(True)
                        res_capacity.append(remain)
                        all_node.remove(all_node[v])
                        break
                    else:
                        remain -= all_node[v].cap
                        all_node[v].cap = 0
                        res.append(all_node[v].id)
                        res_flag.append(True)
                        res_capacity.append(remain)
                        all_node.remove(all_node[v])
            elif node.cap < 5:
                res.append(node.id)
                res_capacity.append(5)
                res_flag.append(False)
                remain = 5 - node.cap % 5
                while remain > 0:
                    if len(all_node) == 0:
                        print("ok2")
                        break
                    v = np.random.randint(0, high=len(all_node), size=None)
                    if all_node[v].cap > remain:
                        all_node[v].cap -= remain
                        res.append(all_node[v].id)
                        res_capacity.append(remain)
                        res_flag.append(True)
                        break
                    elif all_node[v].cap == remain:
                        all_node[v].cap -= remain
                        res.append(all_node[v].id)
                        res_flag.append(True)
                        res_capacity.append(remain)
                        all_node.remove(all_node[v])
                        # if len(all_node) == 0:
                        #     res.append(all_node[0].id)
                        #     res_flag.append(False)
                        #     res_capacity.append(all_node[0].cap)
                        #     return res, res_flag, res_capacity
                        break
                    else:
                        remain -= all_node[v].cap
                        all_node[v].cap = 0
                        res.append(all_node[v].id)
                        res_capacity.append(remain)
                        res_flag.append(True)
                        all_node.remove(all_node[v])
        return res, res_flag, res_capacity

    def route2dis(self, route, route_flag, route_cap):
        dis = 0
        cost = 0
        for i in range(len(route)):
            if not route_flag[i]:
                if i + 1 < len(route) and route_flag[i + 1]:
                    dis += self.distance_array[0][route[i]]
                    cost += (20 + route_cap[i] * 10) * self.distance_array[0][route[i]] / 1000
                else:
                    dis += self.distance_array[0][route[i]]
                    dis += self.distance_array[route[i]][0]
                    cost += (20 + route_cap[i] * 10) * (self.distance_array[0][route[i]] / 1000)
                    cost += (20 + 0 * 10) * (self.distance_array[route[i]][0] / 1000)
            else:
                dis += self.distance_array[route[i - 1]][route[i]]
                cost += (20 + route_cap[i] * 10) * (self.distance_array[route[i - 1]][route[i]] / 1000)
                if (i + 1 < len(route) and route_flag[i + 1] is False) or i == len(route) - 1:
                    dis += self.distance_array[route[i]][0]
                    cost += (20 + 0 * 10) * (self.distance_array[route[i - 1]][route[i]] / 1000)
        return cost

    def run(self):
        start = time.time()
        cost_array = []
        for i in range(self.iter_num):
            temp, temp_flag, temp_cap = self.generate_init_solution()
            new_dis = self.route2dis(temp, temp_flag, temp_cap)
            df = new_dis - self.result_distance
            if df < 0:
                self.result_distance = new_dis
                self.result = temp
                self.result_flag = temp_flag
                self.result_capacity = temp_cap
            elif np.exp(-df / self.T) > np.random.random():
                self.result_distance = new_dis
                self.result = temp
                self.result_flag = temp_flag
                self.result_capacity = temp_cap
            now = time.time()
            if i % 100 == 0:
                print("第" + str(i) + "次迭代：")
                temp = []
                for i in self.result:
                    temp.append(i + 1)
                print("路径: {}".format(temp))
                print("flag: {}".format(self.result_flag))
                print("花费: {}元".format(self.result_distance))
                print("cap: {}".format(self.result_capacity))
                print("运行时间：{}".format(now - start))
                print()
                cost_array.append(self.result_distance)
            self.T = self.alpha * self.T
            if self.T < self.e:
                break
        plt.plot(range(len(cost_array)), cost_array)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.title("运输成本随迭代次数变化情况")
        plt.xlabel("迭代次数(次)")
        plt.ylabel("运输成本(元)")
        plt.show()
        print("最后一次：")
        temp = []
        for i in self.result:
            temp.append(i + 1)
        now = time.time()
        print("路径: {}".format(temp))
        print("flag: {}".format(self.result_flag))
        print("花费: {}元".format(self.result_distance))
        print("cap: {}".format(self.result_capacity))
        print("运行时间：{}".format(now - start))
        print()


class Node:
    def __init__(self, cap, node_id):
        self.cap = cap
        self.id = node_id


if __name__ == '__main__':
    solution = Solution()
    # solution.test()
    solution.run()
