# KNN算法的简单实现，输入为N个二维数组，手动输入需要聚类的数目
# 2019年09月07日

import numpy as np


# from calc_centers import calc_centers


def calc_centers(z, centroids, labels):
    ctrs = np.zeros(centroids.shape)  # 注意坑 $$$$一定用新容器装新变量，严禁引用输入变量，会导致异常修改原数据如： ctrs=centers，会对原centers变量修改
    num = centroids.shape[0]
    for i in range(num):
        index = (labels == i + 1).nonzero()[0]  # 取逻辑索引的技巧，注意行索引需要加[0]
        subset1 = z[index, :]  # 逻辑索引的使用
        ctrs[i, :] = np.mean(subset1, axis=0)  # 平均函数的使用，axis=0是对列求平均
    return ctrs


x = [0, 1, 2, 4, 5, 5, 6, 1, 1, 1]
y = [0, 1, 1, 3, 3, 4, 5, 4, 5, 6]
# x=[0.,1.,0.,1.,2.,1.,2.,3.,6.,7.,8.,6.,7.,8.,9.,7.,8.,9.,8.,9.]
# y=[0.,0.,1.,1.,1.,2.,2.,2.,6.,6.,6.,7.,7.,7.,7.,8.,8.,8.,9.,9.]

z = np.array(list(zip(x, y)))  # 注意    np.array(zip(x,y))不可以用,zip必须先用list解包
z = z.astype(np.float)  # 注意坑  $$$注意数值类型，有时int类型会出错

num = int(input('输入聚类数'))
centers = z[0:num, :]
labels = np.zeros((len(x), 1))
dists = np.zeros((len(x), 1))

label_change = True
while label_change:
    label_change = False

    for i in range(len(x)):
        dist = 1000000
        labelstmp = 0
        for j in range(num):
            dist1 = np.sqrt(np.sum(np.square(centers[j, :] - z[i, :])))
            if dist1 < dist:
                dist = dist1
                labelstmp = j + 1

        dists[i] = dist

        if labels[i] != labelstmp:
            labels[i] = labelstmp
            label_change = True

    centers = calc_centers(z, centers, labels)

print('The Cluster labels are:')
print(labels)
# print('The min dists are:')
# print(dists)
print('The Cluster centers are:')
print(centers)
