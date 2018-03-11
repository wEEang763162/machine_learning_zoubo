#!/usr/bin/python
#  -*- coding:utf-8 -*-

import numpy as np
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


if __name__ == '__main__':
    x1, x2 = np.mgrid[-5:5:51j, -5:5:51j]
    x = np.stack((x1, x2), axis=2)

    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['font.sans-serif'] = 'SimHei'
    plt.figure(figsize=(9, 8), facecolor='w')
    sigma = (np.identity(2), np.diag((3,3)), np.diag((2,5)), np.array(((2,1), (2,5))))
    for i in np.arange(4):
        ax = plt.subplot(2, 2, i+1, projection='3d')
        norm = stats.multivariate_normal((0, 0), sigma[i])
        y = norm.pdf(x)
        ax.plot_surface(x1, x2, y, cmap=cm.Accent, rstride=2, cstride=2, alpha=0.9, lw=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    plt.suptitle('二元高斯分布方差比较', fontsize=18)
    plt.tight_layout(1.5)
    plt.show()
