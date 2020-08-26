import numpy as np
from scipy.misc import derivative
from sympy import diff, symbols
import sympy
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def func(t, index):
    v_max = [0.0535, 0.0605, 0.0552]
    p_max = [35.9016, 38.0052, 41.7935]
    p_0 = 0.3
    t_0 = 70.
    down = 1 - ((p_0 / p_max[index]) * (1 - sympy.exp(v_max[index] * (t - t_0))))
    up = (p_0 * sympy.exp(v_max[index] * (t - t_0)))

    return up / down


def def_fuc(t):
    return 0.000379351846215239 * sympy.exp(0.0535 * t) / (
            0.000197503418073388 * sympy.exp(0.0535 * t) + 0.991643826458988) - 7.49232862799598e-8 * sympy.exp(
        0.107 * t) / (
                   0.000197503418073388 * sympy.exp(0.0535 * t) + 0.991643826458988) ** 2


def def_fuc1(t):
    return 0.000262808555089338 * sympy.exp(0.0605 * t) / (
            0.000114298650167027 * sympy.exp(0.0605 * t) + 0.99210634334249) - 3.00386630990582e-8 * sympy.exp(
        0.121 * t) / (0.000114298650167027 * sympy.exp(0.0605 * t) + 0.99210634334249) ** 2


def def_fuc2(t):
    return 0.000347493315184004 * sympy.exp(0.0552 * t) / (
            0.000150625545918227 * sympy.exp(0.0552 * t) + 0.992821850287724) - 5.23413703025249e-8 * sympy.exp(
        0.1104 * t) / (
                   0.000150625545918227 * sympy.exp(0.0552 * t) + 0.992821850287724) ** 2


# x = symbols("x", real=True)
# print(diff(func(x, 2), x))
n = np.linspace(0, 500, 500)
y = []
dy = []
y1 = []
dy1 = []
y2 = []
dy2 = []
plt.rcParams['font.sans-serif'] = ['SimHei']
for i in n:
    y.append(func(i, 0))
for i in n:
    dy.append(def_fuc(i))

for i in n:
    y1.append(func(i, 1))
for i in n:
    dy1.append(def_fuc1(i))
for i in n:
    y2.append(func(i, 1))
for i in n:
    dy2.append(def_fuc2(i))
plt.subplot(331)
plt.plot(n, y)
plt.title('α 0.1 函数')
plt.subplot(334)
plt.title('α 0.1 导函数')
plt.plot(n, dy)

plt.subplot(332)
plt.plot(n, y1)
plt.title('α 0.4 函数')
plt.subplot(335)
plt.plot(n, dy1)
plt.title('α 0.4 导函数')

plt.subplot(333)
plt.plot(n, y2)
plt.title('α 0.8 函数')
plt.subplot(336)
plt.plot(n, dy2)
plt.title('α 0.8 导函数')

plt.show()

# res = minimize_scalar(func, bounds=(0, 500), method='bounded')
# print(res.x)
# print(func(500))
# print(func(309.0170060628528))
