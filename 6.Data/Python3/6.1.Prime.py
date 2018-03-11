#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from time import time
import math


def is_prime(x):
    return 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)]


def is_prime3(x):
    flag = True
    for p in p_list2:
        if p > math.sqrt(x):
            break
        if x % p == 0:
            flag = False
            break
    if flag:
        p_list2.append(x)
    return flag


if __name__ == "__main__":
    a = 2
    b = 1000

    # 方法1：直接计算
    t = time()
    p = [p for p in range(a, b) if 0 not in [p % d for d in range(2, int(math.sqrt(p)) + 1)]]
    print(time() - t)
    print(p)

    # 方法2：利用filter
    t = time()
    p = list(filter(is_prime, list(range(a, b))))
    print(time() - t)
    print(p)

    # 方法3：利用filter和lambda
    t = time()
    is_prime2 = (lambda x: 0 not in [x % i for i in range(2, int(math.sqrt(x)) + 1)])
    p = list(filter(is_prime2, list(range(a, b))))
    print(time() - t)
    print(p)

    # 方法4：定义
    t = time()
    p_list = []
    for i in range(2, b):
        flag = True
        for p in p_list:
            if p > math.sqrt(i):
                break
            if i % p == 0:
                flag = False
                break
        if flag:
            p_list.append(i)
    print(time() - t)
    print(p_list)

    # 方法5：定义和filter
    p_list2 = []
    t = time()
    list(filter(is_prime3, list(range(2, b))))
    print(time() - t)
    print(p_list2)

    print('---------------------')
    a = 750
    b = 900
    p_list2 = []
    np.set_printoptions(linewidth=150)
    p = np.array(list(filter(is_prime3, list(range(2, b+1)))))
    p = p[p >= a]
    print(p)
    p_rate = float(len(p)) / float(b-a+1)
    print('素数的概率：', p_rate, end='\t  ')
    print('公正赔率：', 1/p_rate)
    print('合数的概率：', 1-p_rate, end='\t  ')
    print('公正赔率：', 1 / (1-p_rate))

    alpha1 = 5.5 * p_rate
    alpha2 = 1.1 * (1 - p_rate)
    print('赔率系数：', alpha1, alpha2)
    print(1 - (alpha1 + alpha2) / 2)
    print((1 - alpha1) * p_rate + (1 - alpha2) * (1 - p_rate))
