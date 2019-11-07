#!/usr/bin/env python
# coding: utf-8
'''
author:wlh19991127
机器学习算法2：逻辑回归 
包含正则化、特征映射

'''


import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#读取文件
def read_txt(filename,split):
    data=np.loadtxt(filename,delimiter=split,dtype=np.float64)
    x=data[:,0:-1]  #数据集
    y=data[:,-1]    #结果集
    return x,y

#画散列图 只能针对两个属性的数据集
def plot_scatt(x,y):
    x1=x[:,0]
    x2=x[:,1]
    for i in range(0,len(y)):
        if y[i]==1:
            plt.scatter(x1[i],x2[i],marker='o',color='green')
        else:
            plt.scatter(x1[i],x2[i],marker='x',color='red')

#S函数，将假设函数映射到[0,1]区间
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#特征映射，将属性映射为更高次幂  power为最高次幂
def feathure_map(x1,x2,power):
    m=x1.shape[0]
    X=np.ones((m,1))
    for i in range(1,power+1):
        for j in range(0,i+1):
            x_temp=(x1**j)*(x2**(i-j))
            X=np.hstack((X,x_temp.reshape((m,1))))
    return X

#计算代价函数，lamda为可选项，指定lamda的值则为正则化
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

#计算梯度，lamda为可选项，指定lamda则为正则化
def gradient(theta,X,y,lamda=0):
    m=len(y)
    h=sigmoid(X@theta)
    theta1=theta.copy()
    theta1[0]=0
    grad=(X.T@(h-y))/m+lamda/m*theta1
    return grad

#预测函数
def predict(x,theta,power=1):
    X=np.hstack((np.ones((1,1)),x))
    X=feathure_map(X[:,1],X[:,2],power)
    pred=sigmoid(X@theta)
    if pred>=0.5:
        print("预测结果为1,准确率为:",pred)
    else:
        print("预测结果为0,准确率为:",1-pred)

#作出决策边界
def decision_boundary(theta,power):
    u = np.linspace(-1,1.5,50)  #根据具体的数据，这里需要调整
    v = np.linspace(-1,1.5,50)
    z = np.zeros((len(u),len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(feathure_map(u[i].reshape(1,1),v[j].reshape(1,1),power),theta)    # 计算对应的值，需要map
    z = np.transpose(z)
    plt.contour(u,v,z,0) 

#运行
def run(lamda=0,power=1):
    x,y=read_txt('C:/Users/wang/Desktop/机器学习编程/machine-learning-ex2/ex2/ex2data2.txt',',')
    m=len(y) #数据的条数 100
    y=y.reshape((m,1))
    plot_scatt(x,y) #画出散点图
    X=np.hstack((np.ones((m,1)),x)) #X矩阵
    X=feathure_map(X[:,1],X[:,2],power)
    args=X.shape[1] #特征值的数量 2
    theta=np.zeros((args,1))
    cost_start=calculate_cost(theta,X,y)
    print("初始代价函数的值为",cost_start)
    
    '''
        fmin_tnc()函数用于求解指定函数的最小值，第一个参数为要求解最小值的函数，第二个参数为初始化的参数，第三个参数为
        梯度函数，第四个参数是一个参数列表 要注意传入的数据类型
    '''
    
    theta=theta.reshape((args))
    y=y.reshape((y.shape[0]))
    result=optimize.fmin_bfgs(calculate_cost,theta,fprime=gradient,args=(X,y,lamda))
    cost_end=calculate_cost(theta,X,y,lamda)
    print("最终的代价函数的值为",cost_end)
    decision_boundary(result,power)
    pred_x=np.array((0.25,0.25)).reshape((1,2))
    predict(pred_x,result,power)
    
if __name__=='__main__':
    run(lamda=1,power=6)
    
    
if __name__=='__main__':
    run(lamda=1,power=6)

