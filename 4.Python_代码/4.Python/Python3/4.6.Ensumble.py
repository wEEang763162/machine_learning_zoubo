#!/usr/bin/python
# -*- coding:utf-8 -*-

import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import reduce


def c(n, k):
    return reduce(operator.mul, list(range(n-k+1, n+1))) / reduce(operator.mul, list(range(1, k+1)))


def bagging(n, p):
    s = 0
    for i in range(n // 2 + 1, n + 1):
        s += c(n, i) * p ** i * (1 - p) ** (n - i)
    return s


if __name__ == "__main__":
    n = 100
    x = np.arange(1, n, 2)
    y = np.empty_like(x, dtype=np.float)
    for i, t in enumerate(x):
        y[i] = bagging(t, 0.6)
        if t % 10 == 9:
            print(t, '次采样正确率：', y[i])
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(x, y, 'ro-', lw=2)
    plt.xlim(0, n)
    plt.ylim(0.6, 1)
    plt.xlabel('采样次数', fontsize=16)
    plt.ylabel('正确率', fontsize=16)
    plt.title('Bagging', fontsize=20)
    plt.grid(b=True)
    plt.show()
