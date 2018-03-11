# -*- coding:utf-8 -*-
# /usr/bin/python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math


def func(a):
    if a < 1e-6:
        return 0
    last = a
    c = a / 2
    while math.fabs(c - last) > 1e-6:
        last = c
        c = (c + a/c) / 2
    return c


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x = np.linspace(0, 30, num=50)
    func_ = np.frompyfunc(func, 1, 1)
    y = func_(x)
    # y = np.sqrt(x)
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.plot(x, y, 'ro-', lw=2, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.title('这段代码在计算什么？', fontsize=18)
    plt.show()
