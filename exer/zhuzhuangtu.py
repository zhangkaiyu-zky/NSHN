# -*- coding: utf-8 -*-
# @Time : 2023/5/6 8:54
# @Author : 张凯玉
# @File : zhuzhuangtu.py
# @Project : SELFRec-main



import matplotlib.pyplot as plt

# 定义数据
#SGL：0.2332 SimGCL：0.2328，LightGCN：0.2230, ours：0.25707
# results = {'LightGCN': 0.2230, 'SGL':0.2332,'SimGCL':0.2328,'SCL':0.2350,'ours':0.25707}#gowalla数据集
# results = {'LightGCN': 0.0639, 'SGL':0.0675, 'SCL': 0.0684,'SimGCL':0.0721,'ours':0.07312} #yelp数据集

results = {'BUIR':0.1127,'Mult-VAE':0.1310,'DNN+SSL':0.1366,'LightGCN': 0.1501, 'MixGCF':0.1731,'SimGCL':0.1772,'ours':0.18018}#douban数据集
# results = {'BUIR':0.0487,'Mult-VAE':0.0584,'DNN+SSL':0.0483,'LightGCN': 0.0639, 'MixGCF':0.0713,'SimGCL':0.0721,'ours':0.07312}#yelp数据集
names = list(results.keys())
values = list(results.values())

# 定义每个数据点的颜色
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple','Pink']

# 绘制柱状图
plt.bar(names, values, color=colors)

# 设置图表标题和标签
plt.title('Comparison of different CL methods Models in Yelp')
plt.xlabel('Models')
plt.ylabel('recall(20)')

# 显示图表
plt.show()





