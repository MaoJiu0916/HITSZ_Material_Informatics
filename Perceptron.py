# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 15:36:36 2021

@author: 35092
"""


from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def matplot():
    plt.figure()
    plt.title("Perceptron",fontsize=15)
    plt.xlabel('X',fontsize=12)
    plt.ylabel('Y',fontsize=12)
    return plt

data1 = pd.read_csv('example2.csv')

x1 = data1.iloc[:,:-1] 
y1 = data1.iloc[:,-1]

clf = Perceptron(fit_intercept=False,max_iter=30,shuffle=False)
clf.fit(x1,y1)
print(clf.coef_,clf.intercept_)

matplot()
x_min, x_max = x1.iloc[:, 0].min() - 1, x1.iloc[:, 0].max() + 1 
y_min, y_max = x1.iloc[:, 1].min() - 1, x1.iloc[:, 1].max() + 1 
XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]

plt.scatter(x1.iloc[0:14, 0], x1.iloc[0:14, 1], marker='*',c='cyan',s=80,\
            edgecolors='k',alpha=0.8,label='BCC') 
plt.scatter(x1.iloc[14:36, 0], x1.iloc[14:36, 1], marker='o',c='yellow',\
            s=80,edgecolors='k',alpha=0.8,label='FCC')
line_x = np.arange(x_min,x_max)
line_y = line_x * (-clf.coef_[0][0] / clf.coef_[0][1]) - clf.intercept_
plt.plot(line_x,line_y,'r--')
plt.show()