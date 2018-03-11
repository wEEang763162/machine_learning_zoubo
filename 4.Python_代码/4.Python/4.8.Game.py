#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy import special

if __name__ == '__main__':
    method = 'simple'

    # 1.暴力模拟
    if method == 'simulation':
        p = 0.6
        a, b, c = 0, 0, 0
        t, T = 0, 1000000
        while t < T:
            a = b = 0
            while (a <= 11) and (b <= 11):
                if np.random.uniform() < p:
                    a += 1
                else:
                    b += 1
            if a > b:
                c += 1
            t += 1
        print float(c) / float(T)
    # 2.直接计算
    elif method == 'simple':
        answer = 0
        p = 0.836435199842     # 每分的胜率
        N = 3      # 每局多少分
        for x in np.arange(N):  # x为对手得分
            answer += special.comb(N + x - 1, x) * ((1-p) ** x) * (p ** N)
        print answer
    # 3.严格计算
    else:
        answer = 0
        p = 0.6  # 每分的胜率
        N = 11  # 每局多少分
        for x in np.arange(N-1):  # x为对手得分：11:9  11:8  11:7  11:6...
            answer += special.comb(N + x - 1, x) * ((1 - p) ** x) * (p ** N)
        p10 = special.comb(2*(N-1), N-1) * ((1-p)*p) ** (N-1)   # 10:10的概率
        t = 0
        for n in np.arange(100):    # {XO}(0,)|OO   思考：可以如何简化？
            t += (2*p*(1-p)) ** n * p * p
        answer += p10 * t
        print answer
