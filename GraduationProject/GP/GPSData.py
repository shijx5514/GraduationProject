# -*- coding: utf-8 -*-
import linecache
import re
from sklearn.preprocessing import Imputer
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

#导入练习数据
def importData(path):
    f = open(path, "r")
    count = len(f.readlines())
    linenum = 5
    first_ele = True
    while linenum <= count:
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
    print martix
    return martix
    f.close()


#缺失值插补
def missingValue (X,Y):
    imp = Imputer(missing_values=0, strategy='mean', axis=0)
    print("缺失值插补")
    print(imp.fit(Y))

    print(imp.transform(X))
    print((X[:,1]).mean())
    print((X[:,2]).mean())
    print((X[:,3]).mean())
    print((X[:, 1]).max())
    print((X[:, 2]).max())
    print((X[:, 3]).max())
    print((X[:, 1]).min())
    print((X[:, 2]).min())
    print((X[:, 3]).min())
def predict(martix):
    x = martix[:, 0]
    y1 = martix[:, 1]
    y2 = martix[:, 2]
    y3 = martix[:, 3]

    reg = linear_model.Ridge(alpha=.5)
    reg.fit(x, y1)
    print(reg.coef_)

    print(reg.intercept_)
    return reg

# 绘图
def printPicture(model, train_martix):
    plt.figure()
    kk = model.coef_[0][0]  # 获得预测模型的参数
    bb = model.intercept_[0]  # 获得预测模型的截距
    plt.title('GPS \n  Predicted:y=' + str(kk)[0:4] + 'x +' + str(bb)[0:4]);
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)  # 显示网格
    x = train_martix[:, 0]
    x.astype(float)
    y1 = train_martix[:, 1]
    yy = model.predict(x)
    plt.plot(x, y1, 'r.')  # 绘图
    plt.plot(x, yy, 'g-')  # 绘图
    plt.show()  # 显示图像

martix = importData("C:/Users/ShiJiaXin/Desktop/data/data/data/GPSData/2015-07-08/12-00-00.txt")
train_martix = importData("C:/Users/ShiJiaXin/Desktop/data/data/data/GPSData/2015-07-08/13-00-00.txt")
missingValue(martix, train_martix)
model = predict(martix)
printPicture(model, train_martix)
