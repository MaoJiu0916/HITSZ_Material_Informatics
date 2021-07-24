# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 14:47:28 2021

@author: 35092
"""


from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np

def matplot():
    plt.figure()
    plt.title("LASSO",fontsize=15)
    plt.xlabel('X',fontsize=12)
    plt.ylabel('Y',fontsize=12)
    return plt

x = [[25],[50],[70],[80]]
y = [[4.11],[4.92],[6.10],[6.66]]

reg1 = linear_model.Lasso(alpha = 0.1)
reg1.fit(x,y)
reg2 = linear_model.Lasso(alpha = 1)
reg2.fit(x,y)
reg3 = linear_model.Lasso(alpha = 10)
reg3.fit(x,y)


x1 = [[20],[40],[60],[90]]
y1 = reg1.predict(x1)
y2 = reg2.predict(x1)
y3 = reg3.predict(x1)

print(reg1.intercept_,reg1.coef_)
print(reg2.intercept_,reg2.coef_)
print(reg3.intercept_,reg3.coef_)

matplot()
plt.scatter(x, y,c='#DC143C',s=np.pi * 4**2,alpha=0.4)
plt.plot(x1,y1,'r--',label='alpha=0.1')
plt.plot(x1,y2,'g--',label='alpha=1')
plt.plot(x1,y3,'b--',label='alpha=10')
plt.legend(loc="best",fontsize=10)
plt.show()