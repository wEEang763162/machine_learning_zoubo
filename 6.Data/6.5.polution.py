#!/usr/bin/python
# -*- encoding: utf-8

import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor


def read_data():
    plt.figure(figsize=(13, 7), facecolor='w')
    plt.subplot(121)
    data = pd.read_csv('C0904.csv', header=0)
    x = data['H2O'].values
    plt.plot(x, 'r-', lw=1, label=u'C0904')
    plt.title(u'实际排放数据0904', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid(b=True)

    plt.subplot(122)
    data = pd.read_csv('C0911.csv', header=0)
    x = data['H2O'].values
    plt.plot(x, 'r-', lw=1, label=u'C0911')
    plt.title(u'实际排放数据0911', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid(b=True)

    plt.tight_layout(2, rect=(0, 0, 1, 0.95))
    plt.suptitle(u'如何找到下图中的异常值', fontsize=20)
    plt.show()


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # read_data()
    data = pd.read_csv('C0911.csv', header=0)   # C0911.csv, C0904.csv
    x = data['H2O'].values
    print x

    width = 500
    delta = 10
    eps = 0.15
    N = len(x)
    p = []
    abnormal = []
    for i in np.arange(0, N-width, delta):
        s = x[i:i+width]
        p.append(np.ptp(s))
        if np.ptp(s) > eps:
            abnormal.append(range(i, i+width))
    abnormal = np.array(abnormal).flatten()
    abnormal = np.unique(abnormal)
    # plt.plot(p, lw=1)
    # plt.grid(b=True)
    # plt.show()

    plt.figure(figsize=(18, 7), facecolor='w')
    plt.subplot(131)
    plt.plot(x, 'r-', lw=1, label=u'原始数据')
    plt.title(u'实际排放数据', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid(b=True)

    plt.subplot(132)
    t = np.arange(N)
    plt.plot(t, x, 'r-', lw=1, label=u'原始数据')
    plt.plot(abnormal, x[abnormal], 'go', markeredgecolor='g', ms=3, label=u'异常值')
    plt.legend(loc='upper right')
    plt.title(u'异常检测', fontsize=18)
    plt.grid(b=True)

    # 预测
    plt.subplot(133)
    select = np.ones(N, dtype=np.bool)
    select[abnormal] = False
    t = np.arange(N)
    dtr = DecisionTreeRegressor(criterion='mse', max_depth=10)
    br = BaggingRegressor(dtr, n_estimators=10, max_samples=0.3)
    br.fit(t[select].reshape(-1, 1), x[select])
    y = br.predict(np.arange(N).reshape(-1, 1))
    y[select] = x[select]
    plt.plot(x, 'g--', lw=1, label=u'原始值')    # 原始值
    plt.plot(y, 'r-', lw=1, label=u'校正值')     # 校正值
    plt.legend(loc='upper right')
    plt.title(u'异常值校正', fontsize=18)
    plt.grid(b=True)

    plt.tight_layout(1.5, rect=(0, 0, 1, 0.95))
    plt.suptitle(u'排污数据的异常值检测与校正', fontsize=22)
    plt.show()
