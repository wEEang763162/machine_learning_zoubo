# -*- coding:utf-8 -*-
# /usr/bin/python

from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def draw_eastasia():
    size = 6500*1000    # 6500km
    plt.figure(figsize=(10, 10), facecolor='w')
    m = Basemap(width=size,height=size, projection='lcc', resolution='c', lat_0=35.5, lon_0=103.3)
    m.drawcoastlines(linewidth=0.3, antialiased=False, color='#202020')     # 海岸线
    m.drawrivers(linewidth=0.05, linestyle='-', color=(0.1, 0.1, 0.1), antialiased=False)  # 河流
    m.drawcountries(linewidth=1, linestyle='-', antialiased=False)         # 国界
    m.drawparallels(np.arange(0, 90, 10), labels=[True, True, False, False])   # 绘制平行线(纬线) [left,right,top,bottom]
    m.drawmeridians(np.arange(0, 360, 15), labels=[False, False, False, True], linewidth=1, dashes=[2,2])   # 绘制子午线
    # m.etopo()   # 地形高程
    m.bluemarble()
    plt.tight_layout(4)
    plt.title(u'东亚及附近区域遥感图', fontsize=21)  # 东亚及附近区域地理地形图
    plt.show()


def draw_america():
    plt.figure(figsize=(10, 10), facecolor='w')
    m = Basemap(width=7000*1000, height=7000*1000, projection='lcc', resolution='c', lat_0=50, lon_0=-107)
    m.drawcoastlines(linewidth=0.3, antialiased=False, color='#303030')  # 海岸线
    m.drawcountries(linewidth=1, linestyle='-', antialiased=False, color='k')  # 国界
    m.drawstates(linewidth=0.5, linestyle='--', antialiased=True, color='k')   # 州界
    m.drawparallels(np.arange(0, 90, 10), labels=[True, True, False, False])  # 绘制平行线(纬线) [left,right,top,bottom]
    m.drawmeridians(np.arange(0, 360, 15), labels=[False, False, False, True], linewidth=1)  # 绘制子午线
    m.bluemarble()  # NASA Blue Marble
    plt.tight_layout(4)
    plt.title(u'北美及附近区域遥感图', fontsize=21)
    plt.show()


if __name__ == '__main__':
    mpl.rcParams[u'axes.unicode_minus'] = False
    mpl.rcParams[u'font.sans-serif'] = u'SimHei'

    draw_eastasia()
    # draw_america()
