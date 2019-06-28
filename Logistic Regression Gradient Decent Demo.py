# -*- coding: utf-8 -*-
"""
Created on Mon Nov 05 11:07:26 2018

@author: wwbin
"""

# Logistic Regression
# we need to choose right convergence rate and learning rate, 很重要

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6)
from sklearn.datasets.samples_generator import make_blobs

(X,y) =  make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.05,random_state=20)
#we need to add 1 to X values , add constant term
X1 = np.c_[np.ones((X.shape[0])),X]

plt.scatter(X1[:,1],X1[:,2],marker='o',c=y)
plt.show()

# random choose initial weights
# choose from a uniform distribution, from 0 to 1
W=np.zeros(X1.shape[1])
W

def sigmoid(x):
    return float(1.0 / float((1.0 + np.exp(-1.0*x))))
# check sigmoid function
sx=range(-10,10)
sy=[]
for i in sx:
    sy.append(sigmoid(i))
plt.plot(sx,sy)
plt.show()

# define predict function
def predict():
    predicted_y=[] 
    # for every x in X1
    for x in X1:        
        logit = x.dot(W) #ϴ0+ϴ1*X
        predicted_y.append(sigmoid(logit)) 
    return np.array(predicted_y)

# define cost function, 这里的error不是loss function
def cost_function(predicted_y):
    # cost function 作用是推导出下面的式子， 用predicted减去true y做error
    error=predicted_y-y
    cf=sum(error)    
    return cf,error

# gradient descent
def gradient_descent(lrate,epochs):    
    total_expected_error=float("inf")
    errorlist=[]
    finalepoch=0
    
    for epoch in range(epochs):
        global W
        
        predictedY=predict() 
        total_error,error = cost_function(predictedY)
        # 推倒后logistic regression gradient update 过程和 linear gregression 相同
        gradient=X1.T.dot(error)/X1.shape[0]
        
        if epoch%100==0:
            errorlist.append(total_error)
            finalepoch+=1
          
        if np.abs(total_expected_error-total_error)< 0.00005:
            return errorlist,finalepoch
            
        total_expected_error=total_error        
        W+=-lrate*gradient            

    return errorlist,finalepoch

total_error,finalepoch=gradient_descent(0.01,50000)
#plotting 
plt.plot(range(finalepoch),total_error)
plt.xlabel("epochs in 10's")
plt.ylabel("error")
plt.show()

yhat= predict() # we get the probablities scores (between 0 and 1)

# transfrom from probability to category
#if the score is above 0.5 lets make it 1 else make it 0
# enumerate 用法
for i,v in enumerate(yhat):
    if v >=0.56: 
        yhat[i]=1
    else:
        yhat[i]=0

yhat.astype(int)

########################################################################
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X1,y)

predict_y=clf.predict(X1)
