# -*- coding: utf-8 -*-
"""
Created on Sun Nov 04 10:37:49 2018

@author: wwbin
"""

# some questions:
# 1)how to choose k for KNN: cross validation
# 2)Normally the distance metric uses in kNN is Euclidean distance. 
#   This distance measure becomes meaningless when the dimension of 
#   the data increase significantly. 
#   We can do dimension reduction in the first step.
# 3)need rescale data




# 1.classification
# classify(X,Y,x), X training data, Y labels, x unknown sample
# pseudocode
# for i =1 to m:
#    compute distance(X_i,x) 
# compute set I containing indices for the k smallest distances d(X_i,x)
# return label y

#Generate isotropic Gaussian blobs for clustering

from sklearn.datasets.samples_generator import make_blobs

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6)
import math
# Counter类的目的是用来跟踪值出现的次数。
# 它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value。
# 计数值可以是任意的Interger（包括0和负数）
from collections import Counter
# generate sample
(X,Y) =  make_blobs(n_samples=50,n_features=2,centers=2,cluster_std=1.95,random_state=50)
# plot training data
# s表示点点的大小，c就是color
plt.scatter(X[:,0],X[:,1],marker='o',c=Y)
plt.show()
# prediction point
prediction_points=[[-2,-4],[-3,-6],[1,0],[6,4],[-6,4]]
prediction_points=np.array(prediction_points)
# 视觉呈现加入的点在哪
plt.scatter(X[:,0],X[:,1],marker='o',c=Y)
plt.scatter(prediction_points[:,0],prediction_points[:,1],marker='o')
plt.show()
# define distance function
def get_eculidean_distance(point,k):
    # axis=1 重要, 分列算, euc_distance returns all distance to X
    euc_distance = np.sqrt(np.sum((X - point)**2 , axis=1))
    #agsort return the index of first k smallest number
    return np.argsort(euc_distance)[0:k]
# predict label
def predict(prediction_points,k):
    points_labels=[]
    for point in prediction_points:
        # get label
        distances=get_eculidean_distance(point,k)        
        results=[]
        for index in distances:
            results.append(Y[index])  
        # use counter to count labels
        # 返回第一个most common的结果
        label=Counter(results).most_common(1)
        points_labels.append([point,label[0][0]])
    return points_labels
# prediction result
results=predict(prediction_points,10)
for result in results:
    print("Point = ",result[0])
    print("Class = ",result[1])
    print()

############################################################################
# testing using the same data,检验程序对错
results=predict(X,1)
predictions=[]
for result in results:
    predictions.append(result[1])
# predictions should equal to Y
print(predictions)
Y
############################################################################
# check accuracy, use training sample
def get_accuracy(predictions):
    # number of inconsistency
    error=np.sum(abs(predictions-Y))
    accuracy=100-(error/len(Y))*100
    return accuracy

#for different K values 
acc=[]
for k in range(1,10):
    results=predict(X,k)
    predictions=[]
    for result in results:
        predictions.append(result[1])
    acc.append([get_accuracy(predictions),k])
#plot
plotx=[]
ploty=[]
for a in acc:
    plotx.append(a[1])
    ploty.append(a[0])
    
plt.plot(plotx,ploty)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()







