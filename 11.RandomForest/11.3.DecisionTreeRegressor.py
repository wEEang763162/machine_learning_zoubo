#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


if __name__ == "__main__":
    N = 100
    x = np.random.rand(N) * 6 - 3     # [-3,3)
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    print y
    x = x.reshape(-1, 1)
    print x

    dt = DecisionTreeRegressor(criterion='mse', max_depth=9)
    dt.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = dt.predict(x_test)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(x, y, 'r*', ms=10, label=u'实际值')
    plt.plot(x_test, y_hat, 'g-', linewidth=2, label=u'预测值')
    plt.legend(loc='upper left')
    plt.xlabel(u'X')
    plt.ylabel(u'Y')
    plt.grid(b=True)
    plt.title(u'决策树回归', fontsize=18)
    plt.tight_layout(2)
    plt.show()

    # 比较决策树的深度影响
    depth = [2, 4, 6, 8, 10]
    clr = 'rgbmy'
    dtr = DecisionTreeRegressor(criterion='mse')
    plt.figure(facecolor='w')
    plt.plot(x, y, 'ko', ms=6, label='Actual')
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    for d, c in zip(depth, clr):
        dtr.set_params(max_depth=d)
        dtr.fit(x, y)
        y_hat = dtr.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=c, linewidth=2, label='Depth=%d' % d)
    plt.legend(loc='upper left')
    plt.xlabel(u'X')
    plt.ylabel(u'Y')
    plt.grid(b=True)
    plt.title(u'决策树回归', fontsize=18)
    plt.tight_layout(2)
    plt.show()
