# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:33:51 2017

@author: Ravil
"""
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x=np.linspace(-5,5,100)[:,None]
y=0.5+0.02*x +3*x**2 +np.random.randn()
y.reshape(len(x),1)
plt.scatter(x,y)
plt.show()
model=LinearRegression()
x_new=np.hstack([x,x**2])
model.fit(x_new,y)
theta=model.coef_,model.intercept_
print(theta)
z=[2,4]
model.predict(z,y)
''''plt.scatter(x,y)

plt.plot(x_t,y_pred)
plt.show()'''