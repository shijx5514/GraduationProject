# -*- coding: utf-8 -*-

import linecache
import re
from sklearn.preprocessing import Imputer
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)


##试图处理分析加速度与钻井平台姿态，三个自由度之间的关系。
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

#导入加速度数据
accelerationMartix = importData("C:/Users/ShiJiaXin/Desktop/example/Data/Data/GPSData/2015-04-11/00-00-00.txt")

