# coding:utf-8

import numpy as np
import random
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def linear_regression_demo(n = 25):
    #模拟一个 y = k * x 的数据集,并做一个线性回归,求解k,并做预测
    #首先随机构造一个近似于y = k * x + b 的数据集
    k = random.random()
    b = random.random() * 1
    x = np.linspace(0,n,n)
    y = [ item * k +(random.random() - 0.5) * k * 5 + b for item in x]
    true_y = [ item * k for item in x]
    #进行一元线性回归
    model = LinearRegression()
    model.fit(np.reshape(x,[len(x),1]), np.reshape(y,[len(y),1]))
    yy = model.predict(np.reshape(x,[len(x),1]))
    #绘图
    plt.figure()
    kk = model.coef_[0][0] # 获得预测模型的参数
    bb = model.intercept_[0] #获得预测模型的截距
    plt.title('MebiuW\'s Scikit-Learn Notes : Linear Regression Demo \n True: y='+str(k)[0:4]+'x +'+str(b)[0:4]+'  Predicted:y='+str(kk)[0:4]+'x +'+str(bb)[0:4] );
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True) # 显示网格
    plt.plot(x,y,'r.') # 绘图
    plt.plot(x,yy,'g-') # 绘图
    plt.show() # 显示图像

linear_regression_demo()