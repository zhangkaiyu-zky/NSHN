# -*- coding: utf-8 -*-
# @Time : 2023/5/17 20:21
# @Author : 张凯玉
# @File : exer5.py
# @Project : SELFRec-main


#对三个数据集上噪声的比例对recall的影响。

import matplotlib.pyplot as plt

x = [0,0.05,0.1,0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5]
y1 = [0.2328, 0.25020,0.25802,0.24829, 0.24773,0.24373, 0.24057,0.23534, 0.2331,0.22661,0.22262]#gowalla数据集，recall
y2 = [0.1506, 0.16169,0.16911,0.16088, 0.15996,0.15581, 0.15431,0.15005, 0.1478,0.14277,0.14111]#gowalla数据集 NDCG


# y1 = [0.1772,0.17631,0.18018,0.17583, 0.17426,0.17562, 0.16855,0.16623, 0.15339,0.14812,0.13778 ]#豆瓣数据集，recall
# y2 = [0.1583,0.15575, 0.1586,0.15548, 0.15539,0.15589, 0.1539,0.15158,0.14401,0.1401,0.1316]#豆瓣数据集 NDCG
#
#
#
# y1 = [0.0721,0.07254, 0.07312,0.07208, 0.07135,0.06943, 0.06556,0.05984,0.0555,0.05487,0.05252 ]#yelp数据集，recall
# y2 = [0.0601,0.05958,0.06052,0.05947, 0.0588,0.05736, 0.05443,0.04943,0.04612,0.04581,0.04321]#yelp数据集 NDCG


plt.plot(x, y1, label='recall',marker='o')
plt.plot(x, y2, label='NDCG',marker='^')
plt.xlabel('x')
plt.ylabel('y')
plt.title('gowalla')
plt.legend()
plt.show()
