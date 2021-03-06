# -*- coding: utf-8 -*-
import linecache
import re
import os
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn import svm
np.set_printoptions(threshold=np.inf)
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

#导入练习数据
def importData(path):
    f = open(path, "r")
    count = len(f.readlines())
    linenum = 5
    first_ele = True
    while linenum <= count:
    #先搞100个试试
    #while linenum <= 50:
        line = linecache.getline(path, linenum)
        line = line.strip('\n')
        str = re.split("      |     ",line)
        #print str
        first_el = True
        for ele in str:
            if first_el:
                h, m, se = ele.strip().split(":")
                x_train = np.array(int(h) * 3600 + int(m) * 60 + int(se))
                first_el = False
            else:
                if ele == "":
                    ele = "0"
                x_train = np.append(x_train, ele)

        #记得使用sklearn.preprocessing import Imputer 对缺失值进行插补，先合并成矩阵
        if first_ele:
            array = x_train
            first_ele = False
        else:
            array = np.vstack((array, x_train))
        linenum = linenum + 1
    martix = np.mat(array)
    martix = martix.astype(float)
    #print martix
    return martix
    f.close()

#数据预处理 将度数和位移缩放到零均值和标准正态分布区间

#处理数据，由于采集设备与计算设备的时间不同，提取相同时间的数据并且将矩阵合并
def dealwithdata(attitude_martix, dip_martix):
    attitude_x = attitude_martix[:, 0]
    dip_x = dip_martix[:,0]
    same_x = np.array([]);
    for x in attitude_x:
        for y in dip_x:
            if x == y:
                same_x = np.append(same_x, x)
    print("相同时间")
    print same_x
    return same_x

#根据相同时间获取arritude数据
def trainAttitudeData(same_x, attitude_martix, scaler1, scaler1_first):
    first_ele = True
    t = 0
    array = np.array([])
    while t < len(same_x):
        i = 0
        while i < len(attitude_martix):
            if (attitude_martix[i, 0] == same_x[t]):
                if first_ele:
                    array = attitude_martix[i]
                    first_ele = False
                else:
                    array = np.vstack((array, attitude_martix[i]))
                i = i + 1
                break
            else :
                i = i + 1
        t = t + 1
    print(array.mean())
    print(array.std())
    if scaler1_first:
        scaler1 = preprocessing.StandardScaler().fit(array)
        scaler1_first = False
        print (False)
    reArray = scaler1.transform(array)
    print("姿态平均值数据")
    print(reArray.mean())
    print("方差数据")
    print(reArray.std())
    return reArray

#根据相同时间获取dip数据
def trainDipData(same_x, dip_martix, scaler2, scaler2_first):
    first_ele = True
    t = 0
    array = np.array([])
    while t < len(same_x):
        i = 0
        while i < len(dip_martix):
            if (dip_martix[i, 0] == same_x[t]):
                if first_ele:
                    array = dip_martix[i]
                    first_ele = False
                else:
                    array = np.vstack((array, dip_martix[i]))
                i = i + 1
                break
            else :
                i = i + 1
        t = t + 1
    print(array.mean())
    print(array.std())
    if scaler2_first:
        scaler2 = preprocessing.StandardScaler().fit(array)
        scaler2_first = False
        print (False)
    reArray = scaler2.transform(array)
    print("倾角平均值数据")
    print(reArray.mean())
    print("方差数据")
    print(reArray.std())
    return reArray

#普通最小二乘法的线性回归模型
def train(model, attitude_train_martix, dip_train_martix):
    if (model is None):
        reg = linear_model.LinearRegression()
    else:
        print ("Model used")
        reg = model
    swingData = attitude_train_martix[:,2].reshape(-1, 1)
    xInclination = dip_train_martix[:,2].reshape((-1, 1))
    #试一下多项式回归
    #quadratic_feaurizer = PolynomialFeatures(degree=2)  # 实例化一个二次多项式特征实例
    #xInclination_train_quadratic = quadratic_feaurizer.fit_transform(xInclination)  # 用二次多项式对样本X值做变换
    reg.fit(xInclination, swingData)

    return reg

#神经网络模型的多层感知器回归器预测
def mlpTrain(model, attitude_train_martix, dip_train_martix):
    if(model is None):
        reg = neural_network.MLPRegressor(activation='logistic', hidden_layer_sizes=(1,1,1))
    else :
        reg = model
        print ("MLPmodel Used")
    swingData = attitude_train_martix[:, 2].reshape(-1, 1)
    xInclination = dip_train_martix[:, 2].reshape(-1, 1)
    reg.fit(xInclination, swingData)
    print (reg)
    return reg

#岭回归
def ridgeTrain(model, attitude_train_martix, dip_train_martix):
    if (model is None):
        reg = linear_model.Ridge(alpha=.5)
    else:
        reg = model
        print ("Ridgemodel Used")
    swingData = attitude_train_martix[:, 2].reshape(-1, 1)
    xInclination = dip_train_martix[:, 2].reshape(-1, 1)
    reg.fit(xInclination, swingData)
    print (reg)
    return reg

#svm支持向量机
def svmTrain(model, attitude_train_martix, dip_train_martix):
    if(model is None) :
        clf = svm.SVR(C = 2.0, kernel='sigmoid')
    else :
        clf = model
        print ("SVMmodel Used")
    swingData = attitude_train_martix[:, 2].reshape(-1, 1)
    xInclination = dip_train_martix[:, 2].reshape(-1, 1)
    clf.fit(xInclination, swingData)
    print (clf)
    return clf
#重复K（10）折交叉验证
def tenFold(attitude_train_martix, dip_train_martix):
    random_state = 12883823
    kf = RepeatedKFold(n_splits = 10,  n_repeats = 2, random_state = random_state)
    #for train, test in kf.split(attitude_martix):
        #print(train)
        #print(test)
#准确率结果
def accuracy(model, attitude_train_martix, dip_train_martix):
    swingData = attitude_train_martix[:, 2].reshape((-1, 1))
    xInclination = dip_train_martix[:, 2].reshape((-1, 1))
    scores = cross_val_score(model, xInclination, swingData, cv=5, scoring='r2')
    #print (swingData)
    #print (xInclination)
    print (scores)
# 绘图
def printPicture(model, train_martix1, train_martix2):
    plt.figure()
    kk = model.coef_[0][0]  # 获得预测模型的参数
    bb = model.intercept_[0]  # 获得预测模型的截距
    plt.title('dip2AndAttitude \n  Predicted:y=' + str(kk)[0:4] + 'x +' + str(bb)[0:4]);
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)  # 显示网格
    x = train_martix1[:, 2].reshape((-1, 1))
    x.astype(float)
    y1 = train_martix2[:, 2].reshape((-1, 1))
    yy = model.predict(x)
    plt.plot(x, y1, 'r.')  # 绘图
    plt.plot(x, yy, 'g-')  # 绘图
    plt.show()  # 显示图像

#绘图
def printPicture1(model, train_martix1, train_martix2):
    plt.figure()
    plt.title('dip2AndAttitude, mlp');
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)  # 显示网格
    x = train_martix1[:, 2].reshape((-1, 1))
    #print (x)
    x.astype(float)
    y1 = train_martix2[:, 2].reshape((-1, 1))
    yy = model.predict(x)
    plt.plot(x, y1, 'r.')  # 绘图
    plt.plot(x, yy, 'bo')  # 绘图
    plt.show()  # 显示图像

#绘图
def printPicture2(model, train_martix1, train_martix2):
    plt.figure()
    plt.title('dip2AndAttitude, svm');
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)  # 显示网格
    x = train_martix1[:, 2].reshape((-1, 1))
    #print (x)
    x.astype(float)
    y1 = train_martix2[:, 2].reshape((-1, 1))
    yy = model.predict(x)
    plt.plot(x, y1, 'r.')  # 绘图
    plt.plot(x, yy, 'bo')  # 绘图
    plt.show()  # 显示图像

#绘图
def printPicture3(model, train_martix1, train_martix2):
    plt.figure()
    plt.title('dip2AndAttitude, ridge');
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)  # 显示网格
    x = train_martix1[:, 2].reshape((-1, 1))
    #print (x)
    x.astype(float)
    y1 = train_martix2[:, 2].reshape((-1, 1))
    yy = model.predict(x)
    plt.plot(x, y1, 'r.')  # 绘图
    plt.plot(x, yy, 'bo')  # 绘图
    plt.show()  # 显示图像

# 保存神经网络模型
def saveMlpTrainModel(model):
    joblib.dump(model, "mlpTrainModel.m")
# 保存线性模型
def saveTrainModel(model):
    joblib.dump(model, "trainModel.m")
#调回线性模型
def callTrainModel():
    if (os.path.exists("trainModel.m")) :
        model = joblib.load("trainModel.m")
    else :
        model = None
    return model
#调回神经网络模型
def callMlpTrainModel():
    if (os.path.exists("mlpTrainModel.m")) :
        model = joblib.load("mlpTrainModel.m")
    else :
        model = None
    return model

# 保存svm
def saveSVMTrainModel(model):
    joblib.dump(model, "SVMtrainModel.m")
#调回svm
def callSVMTrainModel():
    if (os.path.exists("SVMtrainModel.m")) :
        model = joblib.load("SVMtrainModel.m")
    else :
        model = None
    return model
# 保存ridge
def saveRidgeTrainModel(model):
    joblib.dump(model, "RidgetrainModel.m")
#调回svm
def callRidgeTrainModel():
    if (os.path.exists("RidgetrainModel.m")) :
        model = joblib.load("RidgetrainModel.m")
    else :
        model = None
    return model
#训练数据导入


attitude_martix = importData("C:/Users/ShiJiaXin/Desktop/data/data/data/attitudeData/2015-07-10/03-00-00.txt")
dip_martix = importData("C:/Users/ShiJiaXin/Desktop/data/data/data/dipData/dip2/2015-07-10/03-00-00.txt")

same_time = dealwithdata(attitude_martix, dip_martix)
scaler1 = preprocessing.StandardScaler()
scaler2 = preprocessing.StandardScaler()
scaler1_first = True
scaler2_first = True
attitude_train_martix = trainAttitudeData(same_time, attitude_martix, scaler1, scaler1_first)
dip_train_martix = trainDipData(same_time, dip_martix, scaler2, scaler2_first)

#print (attitude_train_martix)
#print(dip_train_martix)

#测试数据导入
attitude_martix1 = importData("C:/Users/ShiJiaXin/Desktop/data/data/data/attitudeData/2015-07-10/04-00-00.txt")
dip_martix1 = importData("C:/Users/ShiJiaXin/Desktop/data/data/data/dipData/dip2/2015-07-10/04-00-00.txt")
same_time1 = dealwithdata(attitude_martix1, dip_martix1)
attitude_train_martix1 = trainAttitudeData(same_time1, attitude_martix1, scaler1, scaler1_first)
dip_train_martix1 = trainDipData(same_time1, dip_martix1, scaler2, scaler2_first)
#简单线型模型的回归预测
saveModel = callTrainModel()
model = train(saveModel, attitude_train_martix, dip_train_martix)
saveTrainModel(model)
#神经网络模型的回归预测
saveMlpModel = callMlpTrainModel();
model1 = mlpTrain(saveMlpModel, attitude_train_martix, dip_train_martix)
saveMlpTrainModel(model1)
#SVM支持向量机
saveSVMModel = callSVMTrainModel()
model2 = svmTrain(saveSVMModel, attitude_train_martix, dip_train_martix)
saveSVMTrainModel(model2)
#岭回归
saveRidgeModel = callRidgeTrainModel();
model3 = ridgeTrain(saveRidgeModel, attitude_train_martix, dip_train_martix)
saveRidgeTrainModel(model3)
printPicture(model, dip_train_martix1, attitude_train_martix1)
printPicture1(model1, dip_train_martix1, attitude_train_martix1)
printPicture2(model2, dip_train_martix1, attitude_train_martix1)
printPicture3(model3, dip_train_martix1, attitude_train_martix1)
print ("线性准确率")
accuracy(model, attitude_train_martix1, dip_train_martix1)
print ("mlp准确率")
accuracy(model1, attitude_train_martix1, dip_train_martix1)
print("svm准确率")
accuracy(model2, attitude_train_martix1, dip_train_martix1)
print ("岭回归准确率")
accuracy(model3, attitude_train_martix1, dip_train_martix1)