# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# gen:999 evo,best dist :423.03012511491613

class Draw(object):
    bound_x = []
    bound_y = []

    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.plt = plt
        self.set_font()

    def draw_line(self, p_from, p_to):
        line1 = [(p_from[0], p_from[1]), (p_to[0], p_to[1])]
        (line1_xs, line1_ys) = zip(*line1)
        self.ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color='blue'))

    # def draw_arrow(self, p_from, p_to):
    #     if p_from.shape[0] != 2 and p_to.shape[0] != 2:
    #         print('error,', p_from, p_to)
    #         return
    #     p_from = list(p_from)
    #     p_to = list(p_to)
    #     self.ax.arrow(p_from[0], p_from[1], p_to[0] - p_from[0], p_to[1] - p_from[1],
    #                   length_includes_head=True,
    #                   head_width=(self.bound_x[1] - self.bound_x[0]) / 100,
    #                   head_length=(self.bound_x[1] - self.bound_x[0]) / 50,
    #                   fc='blue', ec='black')

    def draw_points(self, pointx, pointy):
        self.ax.plot(pointx, pointy, 'ro')

    def set_xybound(self, x_bd, y_bd):
        self.ax.axis([x_bd[0], x_bd[1], y_bd[0], y_bd[1]])

    def draw_text(self, x, y, text, size=8):
        self.ax.text(x, y, text, fontsize=size)

    def set_font(self, ft_style='SimHei'):
        plt.rcParams['font.sans-serif'] = [ft_style]  # 用来正常显示中文标签


class TSP(object):
    citys = np.array([])
    citys_name = np.array([])
    pop_size = 50
    c_rate = 0.7
    m_rate = 0.05
    pop = np.array([])
    fitness = np.array([])
    city_size = -1
    ga_num = 200
    best_dist = 1
    best_gen = []
    dw = Draw()

    def __init__(self, c_rate, m_rate, pop_size, ga_num):
        self.fitness = np.zeros(self.pop_size)
        self.c_rate = c_rate
        self.m_rate = m_rate
        self.pop_size = pop_size
        self.ga_num = ga_num

    def init(self):
        tsp = self
        tsp.load_Citys()
        tsp.pop = tsp.creat_pop(tsp.pop_size)
        tsp.fitness = tsp.get_fitness(tsp.pop)
        tsp.dw.bound_x = [np.min(tsp.citys[:, 0]) - 0.005, np.max(tsp.citys[:, 0]) + 0.005]
        tsp.dw.bound_y = [np.min(tsp.citys[:, 1]) - 0.005, np.max(tsp.citys[:, 1]) + 0.005]
        tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)

    # --------------------------------------
    def creat_pop(self, size):
        pop = []
        for i in range(size):
            gene = np.arange(self.citys.shape[0])
            np.random.shuffle(gene)
            pop.append(gene)

        return np.array(pop)

    def get_fitness(self, pop):
        d = np.array([])
        for i in range(pop.shape[0]):
            gen = pop[i]  # 取其中一条染色体，编码解
            dis = self.gen_distance(gen)
            dis = self.best_dist / dis
            d = np.append(d, dis)  # 求路径长
        return d

    def get_local_fitness(self, gen, i):
        '''
        :param gen:城市路径
        :param i:第i城市
        :return:第i城市的局部适应度
        '''
        di = 0
        fi = 0
        if i == 0:
            di = self.ct_distance(self.citys[gen[0]], self.citys[gen[-1]])
        else:
            di = self.ct_distance(self.citys[gen[i]], self.citys[gen[i - 1]])
        od = []
        for j in range(self.city_size):
            if i != j:
                od.append(self.ct_distance(self.citys[gen[i]], self.citys[gen[i - 1]]))
        mind = np.min(od)
        fi = di - mind
        return fi

    def EO(self, gen):
        local_fitness = []
        for g in range(self.city_size):
            f = self.get_local_fitness(gen, g)
            local_fitness.append(f)
        max_city_i = np.argmax(local_fitness)
        maxgen = np.copy(gen)
        if 1 < max_city_i < self.city_size - 1:
            for j in range(max_city_i):
                maxgen = np.copy(gen)
                jj = max_city_i
                while jj < self.city_size:
                    gen1 = self.exechange_gen(maxgen, j, jj)
                    d = self.gen_distance(maxgen)
                    d1 = self.gen_distance(gen1)
                    if d > d1:
                        maxgen = gen1[:]
                    jj += 1
        gen = maxgen
        return gen

    # -------------------------------------
    def select_pop(self, pop):
        best_f_index = np.argmax(self.fitness)
        av = np.median(self.fitness, axis=0)
        for i in range(self.pop_size):
            if i != best_f_index and self.fitness[i] < av:
                pi = self.cross(pop[best_f_index], pop[i])
                pi = self.mutate(pi)
                # d1 = self.distance(pi)
                # d2 = self.distance(pop[i])
                # if d1 < d2:
                pop[i, :] = pi[:]

        return pop

    def select_pop2(self, pop):
        probility = self.fitness / self.fitness.sum()
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=probility)
        n_pop = pop[idx, :]
        return n_pop

    def cross(self, parent1, parent2):
        """交叉"""
        if np.random.rand() > self.c_rate:
            return parent1
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        tempGene = parent2[index1:index2]  # 交叉的基因片段
        newGene = []
        p1len = 0
        for g in parent1:
            if p1len == index1:
                newGene.extend(tempGene)  # 插入基因片段
            if g not in tempGene:
                newGene.append(g)
            p1len += 1
        newGene = np.array(newGene)

        if newGene.shape[0] != self.city_size:
            print('c error')
            return self.creat_pop(1)
            # return parent1
        return newGene

    def mutate(self, gene):
        """突变"""
        if np.random.rand() > self.m_rate:
            return gene
        index1 = np.random.randint(0, self.city_size - 1)
        index2 = np.random.randint(index1, self.city_size - 1)
        newGene = self.reverse_gen(gene, index1, index2)
        if newGene.shape[0] != self.city_size:
            print('m error')
            return self.creat_pop(1)
        return newGene

    def reverse_gen(self, gen, i, j):
        if i >= j:
            return gen
        if j > self.city_size - 1:
            return gen
        parent1 = np.copy(gen)
        tempGene = parent1[i:j]
        newGene = []
        p1len = 0
        for g in parent1:
            if p1len == i:
                newGene.extend(tempGene[::-1])  # 插入基因片段
            if g not in tempGene:
                newGene.append(g)
            p1len += 1
        return np.array(newGene)

    def exechange_gen(self, gen, i, j):
        c = gen[j]
        gen[j] = gen[i]
        gen[i] = c
        return gen

    def evolution(self):
        tsp = self
        for i in range(self.ga_num):
            best_f_index = np.argmax(tsp.fitness)
            worst_f_index = np.argmin(tsp.fitness)
            local_best_gen = tsp.pop[best_f_index]
            local_best_dist = tsp.gen_distance(local_best_gen)
            if i == 0:
                tsp.best_gen = local_best_gen
                tsp.best_dist = tsp.gen_distance(local_best_gen)

            if local_best_dist < tsp.best_dist:
                tsp.best_dist = local_best_dist
                tsp.best_gen = local_best_gen
                # tsp.dw.ax.cla()
                # tsp.re_draw()
                # tsp.dw.plt.pause(0.001)
            else:
                tsp.pop[worst_f_index] = self.best_gen
            print('gen:%d evo,best dist :%s' % (i, self.best_dist))

            tsp.pop = tsp.select_pop(tsp.pop)
            tsp.fitness = tsp.get_fitness(tsp.pop)
            for j in range(self.pop_size):
                r = np.random.randint(0, self.pop_size - 1)
                if j != r:
                    tsp.pop[j] = tsp.cross(tsp.pop[j], tsp.pop[r])
                    tsp.pop[j] = tsp.mutate(tsp.pop[j])
            # self.best_gen = self.EO(self.best_gen)
            tsp.best_dist = tsp.gen_distance(self.best_gen)

    def load_Citys(self, file='china.csv', delm=';'):
        # 中国34城市经纬度
        # data = pd.read_csv(file, delimiter=delm, header=None).values
        path = "data/B题附件1.xlsx"
        data = pd.read_excel(path, sheet_name=0).values
        self.citys = data[:, 1:]
        self.citys_name = list(range(30))
        self.citys_name[0] = 'data center'
        self.city_size = data.shape[0]

    def gen_distance(self, gen):
        distance = 0.0
        for i in range(-1, len(self.citys) - 1):
            index1, index2 = gen[i], gen[i + 1]
            city1, city2 = self.citys[index1], self.citys[index2]
            distance += self.ct_distance(city1, city2)
            # distance += np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
        return distance

    def ct_distance(self, x, y):
        # x,y分别表示一个经纬度坐标点
        return 6370 * math.acos(math.cos(x[0] - y[0]) * math.cos(x[1]) * math.cos(y[1]) + math.sin(x[1]) * math.sin(y[1]))
    # def ct_distance(self, x1, y1, x2, y2):
    #     # x,y分别表示一个经纬度坐标点
    #     if x1 == x2 and y1 == y2:
    #         return 0
    #     return 6370 * math.acos(math.cos(x1 - x2) * math.cos(y1) * math.cos(y2) + math.sin(y1) * math.sin(y2))

    # def ct_distance(self, city1, city2):
    #     d = np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
    #     return d

    def draw_citys_way(self, gen):
        '''
        根据一条基因gen绘制一条旅行路线
        :param gen:
        :return:
        '''
        tsp = self
        dw = self.dw
        m = gen.shape[0]
        tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)
        for i in range(m):
            if i < m - 1:
                best_i = tsp.best_gen[i]
                next_best_i = tsp.best_gen[i + 1]
                best_icity = tsp.citys[best_i]
                next_best_icity = tsp.citys[next_best_i]
                dw.draw_line(best_icity, next_best_icity)
        start = tsp.citys[tsp.best_gen[0]]
        end = tsp.citys[tsp.best_gen[-1]]
        dw.draw_line(end, start)

    def draw_citys_name(self, gen, size=5):
        '''
        根据一条基因gen绘制对应城市名称
        :param gen:
        :param size: text size
        :return:
        '''
        tsp = self
        m = gen.shape[0]
        tsp.dw.set_xybound(tsp.dw.bound_x, tsp.dw.bound_y)
        for i in range(m):
            c = gen[i]
            best_icity = tsp.citys[c]
            tsp.dw.draw_text(best_icity[0], best_icity[1], tsp.citys_name[c], 10)

    def re_draw(self):
        tsp = self
        tsp.dw.draw_points(tsp.citys[:, 0], tsp.citys[:, 1])
        tsp.draw_citys_name(tsp.pop[0], 8)
        tsp.draw_citys_way(self.best_gen)


def main():
    tsp = TSP(0.5, 0.1, 100, 1000)
    tsp.init()
    tsp.evolution()
    tsp.re_draw()
    tsp.dw.plt.show()


if __name__ == '__main__':
    main()
