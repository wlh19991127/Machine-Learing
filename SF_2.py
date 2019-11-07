#!/usr/bin/env python
# coding: utf-8

# In[214]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def read_txt(filename,split):
    data=np.loadtxt(filename,delimiter=split,dtype=np.float64)
    x=data[:,0:-1]
    y=data[:,-1]
    return x,y

def plot_scatt(x,y):
    x1=x[:,0]
    x2=x[:,1]
    for i in range(0,len(y)):
        if y[i]==1:
            plt.scatter(x1[i],x2[i],marker='o',color='green')
        else:
            plt.scatter(x1[i],x2[i],marker='x',color='red')

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def feathure_map(x1,x2,power):
    m=x1.shape[0]
    X=np.ones((m,1))
    for i in range(1,power+1):
        for j in range(0,i+1):
            x_temp=(x1**j)*(x2**(i-j))
            X=np.hstack((X,x_temp.reshape((m,1))))
    return X

def calculate_cost(theta,X,y,lamda=0):
    j=0
    m=len(y)
    h=sigmoid(X@theta) #假设函数
    first=y.T@np.log(h+1e-5)
    second=(1-y).T@np.log((1-h)+1e-5)
    theta1=theta.copy()
    theta1[0]=0
    j=(first+second)/(-m)+(lamda/(2*m))*theta1.T@theta1
    return j

def gradient(theta,X,y,lamda=0):
    m=len(y)
    h=sigmoid(X@theta)
    theta1=theta.copy()
    theta1[0]=0
    grad=(X.T@(h-y))/m+lamda/m*theta1
    return grad

def predict(x,theta):
    X=np.hstack((np.ones((1,1)),x))
    X=feathure_map(X[:,1],X[:,2],6)
    return sigmoid(X@theta)

def decision_boundary(theta):
   # u=np.linspace(-1,1.5,50)
   # v=np.linspace(-1,1.5,50)
  #  z=np.zeros((len(u),len(v)))
   # for i in range(len(u)):
   #     for j in range(len(v)):
            #print(i,j)
    #        z[i][j]=feathure_map(u[i].reshape(1,1),v[i].reshape(1,1),6)@theta
  #  z=z.T
   # plt.contour(u,v,z,0)
    u = np.linspace(-1,1.5,50)  #根据具体的数据，这里需要调整
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(feathure_map(u[i].reshape(1,-1),v[j].reshape(1,-1),10),theta)    # 计算对应的值，需要map
    z = np.transpose(z)
    plt.contour(u,v,z,0) 

def run():
    x,y=read_txt('C:/Users/wang/Desktop/机器学习编程/machine-learning-ex2/ex2/ex2data2.txt',',')
    m=len(y) #数据的条数 100
    y=y.reshape((m,1))
    plot_scatt(x,y) #画出散点图
    X=np.hstack((np.ones((m,1)),x)) #X矩阵
    X=feathure_map(X[:,1],X[:,2],10)
    args=X.shape[1] #特征值的数量 2
    theta=np.zeros((args,1))
    #print(theta)
    cost_start=calculate_cost(theta,X,y)
    alpha=0.01
   # print(gradient(theta,X,y))
    print("初始代价函数的值为",cost_start)
   # result=optimize.fmin_bfgs(f=calculate_cost,x0=theta,fprime=gradient,args=(X,y))
    #print(result)
    print(X.shape,theta.shape,y.shape)
    theta=theta.reshape((args))
    y=y.reshape((y.shape[0]))
    #result =optimize.fmin_tnc(func=calculate_cost,x0=theta,fprime=gradient,args=(X,y))
    result=optimize.fmin_bfgs(calculate_cost,theta,fprime=gradient,args=(X,y,1))
    #print(result)
    #print(predict(np.array([0.25,0.25]).reshape(1,2),result))
    #X_fm=feathure_map(X[:,1],X[:,2],6)
    decision_boundary(result)
    
    
if __name__=='__main__':
    run()

