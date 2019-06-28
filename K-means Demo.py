# -*- coding: utf-8 -*-
"""
Created on Sun Nov 04 12:08:55 2018

@author: wwbin
"""

# some questions:
# unsupervied algorithms, different from KNN
# 1)how to choose k for KNN: Elbow Method, We run the algorithm for different values of K
#   and plot the K values against SSE(Sum of Squared Errors)
#   SSE=\sum_{1}{k}\sum_{x in C_i} dist^2(m_i,x)
#   distortions.append(sum(np.min(cdist(kmeans_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / kmeans_train.shape[0])

# 2)Normally the distance metric uses in kNN is Euclidean distance. 
#   This distance measure becomes meaningless when the dimension of 
#   the data increase significantly. 
#   We can do dimension reduction in the first step.
# 3)need rescale data


# k-means algorithm:
# 1 - Pick K random points as cluster centers called centroids.
# 2 - Assign each x_i to nearest cluster by calculating its distance to each centroid.
# 3 - Find new cluster center by taking the average of the assigned points.
# 4 - Repeat Step 2 and 3 until none of the cluster assignments change.

from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# we can use ggplot style
plt.style.use('ggplot')
data=pd.read_csv('E:/project/algorithm implementation/xclara_kmeans.csv')

# plot
# 定义X,改数据格式
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
plt.scatter(f1, f2, c='black', s=7)

# calculate distance
# use norm function
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# Number of clusters
k = 3
# 生成在半开半闭区间[low,high)上离散均匀分布的整数值
# X coordinates of random centroids
C_x = np.random.randint(0, 80, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(-20, 70, size=k)
C = np.array(zip(C_x, C_y), dtype=np.float32)
print(C)

# Plotting along with the Centroids
plt.scatter(f1, f2, c='black', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

# To store the value of centroids when it updates
# np.zeros维数写法
C_old = np.zeros(C.shape)
# store Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
# convergence rate
epsilon=0.001
# Loop will run till the error becomes zero
while error > epsilon:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old,None)
#plot
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')



