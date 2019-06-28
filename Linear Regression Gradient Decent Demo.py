# -*- coding: utf-8 -*-
"""
Created on Sun Nov 04 22:26:17 2018

@author: wwbin
"""

# linear regression gradient decent
import numpy as np
import matplotlib.pyplot as plt

#Getting the dataset
points=100
# linspace 用法
X=np.linspace(-3,3,points)
np.random.seed(6)
y=np.sin(X)+np.random.uniform(-0.5,0.5,points)

p=plt.plot(X,y,'ro')
plt.axis([-4,4,-2.0,2.0])
plt.show()

#this concatenates 1 to all X values ex:[1,-3],[1,-2.87],etc... 
# 1 stands for constant
X2=np.c_[np.ones(len(X)),X] 
print(X2.shape)
# iniate start value
W=np.random.uniform(size=X2.shape[1],)
print(W)
# predicted value
def predict():
    return X2.dot(W)
# predicted value
predictedY=predict()

# gradient descent
# lrate is learning rate
def GD(lrate,epochs):
    total_expected_error=0
    errorlist=[]
    finalepoch=0
    for i in range(epochs):
        #必须定义W,不然方程里找不到
        global W
        
        predictedY=predict() 
        error=(predictedY-y)
        total_error=np.sum(error)
        # d/dϴ =error*x.T (transpose to get a vector as result from matrix multiplication)
        # X.shape[0]=size of X (m in GD equation)
        # 不同于矩阵，这里是每个 beta 的 value 有一个 gradient 
        gradient=X2.T.dot(error)/X2.shape[0]        
        # 每100个算一个epoch,没必要每次error都记录
        # 100的原因可能是sample size 是100
        if i%100==0:
            errorlist.append(total_error)
            finalepoch+=1
        #to break when we reach the minimum or avoid overshooting weights, error break
        if np.abs(total_expected_error-total_error)< 0.0005: 
            return errorlist,finalepoch
        
        total_expected_error=total_error        
        W+=-lrate*gradient #new weight=old weight-alpha*change weight
    return errorlist,finalepoch

total_error,finalepoch=GD(0.001,50000)

#plotting 
plt.plot(range(finalepoch),total_error)
plt.xlabel("epochs in 100's")
plt.ylabel("error")
plt.show()

W
plt.plot(predictedY)
