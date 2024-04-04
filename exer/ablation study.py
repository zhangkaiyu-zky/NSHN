# -*- coding: utf-8 -*-
# @Time : 2023/4/21 11:03
# @Author : 张凯玉
# @File : ablation study.py
# @Project : SELFRec-main
import matplotlib.pyplot as plt

# 定义epoch和recall列表
# epoch_list = [1, 2, 3, 4, 5]
# recall_list = [0.85, 0.87, 0.90, 0.92, 0.94]
#
# # 绘制recall随着epoch变化的曲线
# plt.plot(epoch_list, recall_list)
#
# # 添加标题和标签
# plt.title('Recall vs Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Recall')
#
# # 显示图形
# plt.show()
import matplotlib.pyplot as plt

# 定义epoch列表
# epoch_list = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
epoch_list=[]
for i in range(100):
    epoch_list.append(i+1)
print(epoch_list)
# 定义recall列表
# recall_list_1 = [0.85, 0.87, 0.90, 0.92, 0.94]
# recall_list_2 = [0.80, 0.82, 0.85, 0.88, 0.90]
# recall_list_3 = [0.78, 0.80, 0.83, 0.86, 0.88]

# list1=[]
# with open('output.txt','r') as output:
#     lines = output.readlines()
# # print(lines)
# for item in lines:
#     list1.append(eval(item))
#
#
#
#
# recall_list_1 = list1[0]
# recall_list_2 = list1[1]
# recall_list_3 = list1[2]



# 绘制recall随着epoch变化的曲线
recall_list_1=[0.19588, 0.19099, 0.18247, 0.17125, 0.16136, 0.15346, 0.15119, 0.15492, 0.16374, 0.17417, 0.18639, 0.19721, 0.20649, 0.21364, 0.21999, 0.2261, 0.23104, 0.23507, 0.23883, 0.24134, 0.24363, 0.24588, 0.24686, 0.24801, 0.24879, 0.24899, 0.24935, 0.2496, 0.24981, 0.24973, 0.24981, 0.2499, 0.2491, 0.2485, 0.24789, 0.24767, 0.24712, 0.24625, 0.24553, 0.24489, 0.24426, 0.24333, 0.24234, 0.24137, 0.24042, 0.23974, 0.23851, 0.23705, 0.23622, 0.23452, 0.23353, 0.23174, 0.23016, 0.22893, 0.22749, 0.22605, 0.22462, 0.22328, 0.22227, 0.22032, 0.21879, 0.21695, 0.21525, 0.2134, 0.21152, 0.20984, 0.20816, 0.20614, 0.20439, 0.20288, 0.20104, 0.19945, 0.19781, 0.19659, 0.19469, 0.19335, 0.19183, 0.18995, 0.18856, 0.18678, 0.18517, 0.1839, 0.18207, 0.18086, 0.17981, 0.17828, 0.17704, 0.17549, 0.17413, 0.17254, 0.17119, 0.17016, 0.16884, 0.16794, 0.16634, 0.16545, 0.16421, 0.16339, 0.16227, 0.16122]
recall_list_2=[0.19608, 0.19105, 0.18417, 0.1758, 0.1658, 0.15799, 0.15421, 0.15533, 0.16039, 0.16848, 0.17941, 0.19066, 0.20135, 0.21017, 0.21818, 0.22442, 0.22957, 0.2338, 0.23775, 0.2417, 0.24432, 0.24636, 0.24838, 0.24964, 0.24994, 0.25104, 0.25109, 0.25107, 0.25055, 0.25054, 0.24983, 0.24908, 0.24815, 0.24784, 0.24698, 0.24636, 0.24547, 0.24427, 0.24277, 0.24066, 0.23957, 0.23813, 0.23643, 0.23528, 0.23381, 0.23232, 0.23102, 0.22908, 0.22711, 0.22557, 0.22358, 0.22188, 0.2202, 0.21843, 0.21648, 0.21409, 0.21256, 0.21084, 0.20918, 0.20747, 0.20547, 0.20386, 0.20217, 0.20014, 0.19822, 0.19661, 0.19527, 0.19385, 0.19175, 0.18991, 0.18873, 0.18613, 0.18477, 0.18344, 0.18192, 0.18046, 0.17897, 0.17714, 0.17552, 0.17444, 0.17292, 0.17155, 0.17036, 0.16933, 0.16791, 0.1665, 0.16531, 0.16376, 0.1628, 0.16162, 0.16022, 0.15882, 0.158, 0.15675, 0.15563, 0.15451, 0.15372, 0.1526, 0.15164, 0.15049]
recall_list_3=[0.20161, 0.19483, 0.18057, 0.16898, 0.16813, 0.17685, 0.1906, 0.20405, 0.21443, 0.22259, 0.23015, 0.23735, 0.24275, 0.24747, 0.25034, 0.25245, 0.25432, 0.25539, 0.25623, 0.2571, 0.2575, 0.25763, 0.25802, 0.25829, 0.25822, 0.2583, 0.25846, 0.25777, 0.25738, 0.25671, 0.2561, 0.25513, 0.25398, 0.25331, 0.25199, 0.25033, 0.24888, 0.24801, 0.24688, 0.24539, 0.24357, 0.24204, 0.24091, 0.23927, 0.23768, 0.23649, 0.23453, 0.23278, 0.23153, 0.22993, 0.22766, 0.2262, 0.22448, 0.22283, 0.22156, 0.21915, 0.21733, 0.21566, 0.21409, 0.21204, 0.21061, 0.20853, 0.20688, 0.20535, 0.20392, 0.20215, 0.19978, 0.19834, 0.19598, 0.19478, 0.19296, 0.1911, 0.18946, 0.18749, 0.18639, 0.18462, 0.18276, 0.18152, 0.18024, 0.17842, 0.17736, 0.17609, 0.17475, 0.17371, 0.17241, 0.17119, 0.16996, 0.16845, 0.16718, 0.16564, 0.16452, 0.16349, 0.16275, 0.16142, 0.16043, 0.15927, 0.158, 0.15694, 0.15597, 0.15507]



plt.plot(epoch_list, recall_list_1, label='noise ')
plt.plot(epoch_list, recall_list_2, label='negative sampler')
plt.plot(epoch_list, recall_list_3, label='nosie and negative sampler')

# 添加标题和标签
plt.title('Recall vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Recall')

# 添加图例
plt.legend()

# 显示图形
plt.show()




'''
去除负采样：[0.19588, 0.19099, 0.18247, 0.17125, 0.16136, 0.15346, 0.15119, 0.15492, 0.16374, 0.17417, 0.18639, 0.19721, 0.20649, 0.21364, 0.21999, 0.2261, 0.23104, 0.23507, 0.23883, 0.24134, 0.24363, 0.24588, 0.24686, 0.24801, 0.24879, 0.24899, 0.24935, 0.2496, 0.24981, 0.24973, 0.24981, 0.2499, 0.2491, 0.2485, 0.24789, 0.24767, 0.24712, 0.24625, 0.24553, 0.24489, 0.24426, 0.24333, 0.24234, 0.24137, 0.24042, 0.23974, 0.23851, 0.23705, 0.23622, 0.23452, 0.23353, 0.23174, 0.23016, 0.22893, 0.22749, 0.22605, 0.22462, 0.22328, 0.22227, 0.22032, 0.21879, 0.21695, 0.21525, 0.2134, 0.21152, 0.20984, 0.20816, 0.20614, 0.20439, 0.20288, 0.20104, 0.19945, 0.19781, 0.19659, 0.19469, 0.19335, 0.19183, 0.18995, 0.18856, 0.18678, 0.18517, 0.1839, 0.18207, 0.18086, 0.17981, 0.17828, 0.17704, 0.17549, 0.17413, 0.17254, 0.17119, 0.17016, 0.16884, 0.16794, 0.16634, 0.16545, 0.16421, 0.16339, 0.16227, 0.16122]
去除噪声：[0.19608, 0.19105, 0.18417, 0.1758, 0.1658, 0.15799, 0.15421, 0.15533, 0.16039, 0.16848, 0.17941, 0.19066, 0.20135, 0.21017, 0.21818, 0.22442, 0.22957, 0.2338, 0.23775, 0.2417, 0.24432, 0.24636, 0.24838, 0.24964, 0.24994, 0.25104, 0.25109, 0.25107, 0.25055, 0.25054, 0.24983, 0.24908, 0.24815, 0.24784, 0.24698, 0.24636, 0.24547, 0.24427, 0.24277, 0.24066, 0.23957, 0.23813, 0.23643, 0.23528, 0.23381, 0.23232, 0.23102, 0.22908, 0.22711, 0.22557, 0.22358, 0.22188, 0.2202, 0.21843, 0.21648, 0.21409, 0.21256, 0.21084, 0.20918, 0.20747, 0.20547, 0.20386, 0.20217, 0.20014, 0.19822, 0.19661, 0.19527, 0.19385, 0.19175, 0.18991, 0.18873, 0.18613, 0.18477, 0.18344, 0.18192, 0.18046, 0.17897, 0.17714, 0.17552, 0.17444, 0.17292, 0.17155, 0.17036, 0.16933, 0.16791, 0.1665, 0.16531, 0.16376, 0.1628, 0.16162, 0.16022, 0.15882, 0.158, 0.15675, 0.15563, 0.15451, 0.15372, 0.1526, 0.15164, 0.15049]
均使用的时候：[0.20161, 0.19483, 0.18057, 0.16898, 0.16813, 0.17685, 0.1906, 0.20405, 0.21443, 0.22259, 0.23015, 0.23735, 0.24275, 0.24747, 0.25034, 0.25245, 0.25432, 0.25539, 0.25623, 0.2571, 0.2575, 0.25763, 0.25802, 0.25829, 0.25822, 0.2583, 0.25846, 0.25777, 0.25738, 0.25671, 0.2561, 0.25513, 0.25398, 0.25331, 0.25199, 0.25033, 0.24888, 0.24801, 0.24688, 0.24539, 0.24357, 0.24204, 0.24091, 0.23927, 0.23768, 0.23649, 0.23453, 0.23278, 0.23153, 0.22993, 0.22766, 0.2262, 0.22448, 0.22283, 0.22156, 0.21915, 0.21733, 0.21566, 0.21409, 0.21204, 0.21061, 0.20853, 0.20688, 0.20535, 0.20392, 0.20215, 0.19978, 0.19834, 0.19598, 0.19478, 0.19296, 0.1911, 0.18946, 0.18749, 0.18639, 0.18462, 0.18276, 0.18152, 0.18024, 0.17842, 0.17736, 0.17609, 0.17475, 0.17371, 0.17241, 0.17119, 0.16996, 0.16845, 0.16718, 0.16564, 0.16452, 0.16349, 0.16275, 0.16142, 0.16043, 0.15927, 0.158, 0.15694, 0.15597, 0.15507]




'''


