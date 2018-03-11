#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint


if __name__ == "__main__":
    path = 'Advertising.csv'
    # # 手写读取数据
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()#字符串删除空白符
    #     if not d:
    #         continue
    #     d = map(float, d.split(','))#字符串拿出来转化为float
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # # print x
    # # pprint(x)
    # # print y
    # # pprint(y)
    #
    # x = np.array(x)
    # y = np.array(y)
    # print x
    # print y
    #

    # Python自带库
    # f = file(path, 'r')
    # print f
    # d = csv.reader(f)
    # print d,type(d)
    # for line in d:
    #     print line
    # f.close()

    # # numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p
    # print '\n\n===============\n\n'

    # pandas读入
    data = pd.read_csv(path)    # DataFrame,TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    # print x,type(x)
    # print y,type(y)

    mpl.rcParams['font.sans-serif'] = [u'simHei']#在图形框里写汉字
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制1
    # plt.figure(facecolor='w')
    # plt.plot(data['TV'], y, 'ro', label='TV')
    # plt.plot(data['Radio'], y, 'g^', label='Radio')
    # plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    # plt.legend(loc='lower right')
    # plt.xlabel(u'广告花费', fontsize=16)
    # plt.ylabel(u'销售额', fontsize=16)
    # plt.title(u'广告花费与销售额对比数据', fontsize=20)
    # plt.grid()
    # plt.show()

    # # 绘制2
    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()
    #
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print type(x_test),type(y_test)#y_test的类型为<class 'pandas.core.series.Series'>
    print x_train.shape, y_train.shape
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print model
    print linreg.coef_, linreg.intercept_  # 回归系数，截距
    #
    order = y_test.argsort(axis=0)  #将 y_test升序，并返回各元素的原始index,y_test自身不变
    print order,type(order),order.shape  #order的类型为<class 'pandas.core.series.Series'>
    print  y_test.values,y_test.values.shape
    print  y_test.values[order]
    y_test = y_test.values#[order]  # y_test的类型改为<type 'numpy.ndarray'>,一维数组
    print type(y_test)
    print x_test.values
    x_test = x_test.values#[order, :]  # x_test改为二维数组
    print x_test,type(x_test)
    y_hat = linreg.predict(x_test)#linreg.predict返回<type 'numpy.ndarray'>
    print type(y_hat)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print 'MSE = ', mse,
    print 'RMSE = ', rmse
    print 'R2 = ', linreg.score(x_train, y_train)#平均正确率？
    print 'R2 = ', linreg.score(x_test, y_test)

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.grid(b=True)
    plt.show()
