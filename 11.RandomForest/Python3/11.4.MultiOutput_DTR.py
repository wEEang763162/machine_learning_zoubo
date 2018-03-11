#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


if __name__ == "__main__":
    N = 400
    x = np.random.rand(N) * 8 - 4     # [-4,4)
    x = np.random.rand(N) * 4*np.pi     # [-4,4)
    x.sort()
    print(x)
    print('====================')
    # y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    # y2 = np.cos(0.3*x) + np.random.randn(N) * 0.01
    # y1 = np.sin(x) + np.random.randn(N) * 0.05
    # y2 = np.cos(x) + np.random.randn(N) * 0.1
    y1 = 16 * np.sin(x) ** 3 + np.random.randn(N)*0.5
    y2 = 13 * np.cos(x) - 5 * np.cos(2*x) - 2 * np.cos(3*x) - np.cos(4*x) + np.random.randn(N)*0.5
    np.set_printoptions(suppress=True)
    print(y1)
    print(y2)
    y = np.vstack((y1, y2)).T
    print(y)
    print('Data = \n', np.vstack((x, y1, y2)).T)
    print('=================')
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的

    deep = 10
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    dt = reg.fit(x, y)

    x_test = np.linspace(x.min(), x.max(), num=1000).reshape(-1, 1)
    print(x_test)
    y_hat = dt.predict(x_test)
    print(y_hat)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.scatter(y[:, 0], y[:, 1], c='r', marker='s', edgecolor='k', s=60, label='真实值', alpha=0.8)
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='o', edgecolor='k', edgecolors='g', s=30, label='预测值', alpha=0.8)
    plt.legend(loc='lower left', fancybox=True, fontsize=12)
    plt.xlabel('$Y_1$', fontsize=12)
    plt.ylabel('$Y_2$', fontsize=12)
    plt.grid(b=True, ls=':', color='#606060')
    plt.title('决策树多输出回归', fontsize=15)
    plt.tight_layout(1)
    plt.show()
