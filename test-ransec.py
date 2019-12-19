import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
n_samples = 1000
n_outliers = 50
x, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)
a = np.arange(15).reshape(3, 5)
print(a)

import math

df = pd.read_csv('scanData.txt',delimiter=',')
x = df.values[:,0]
y = df.values[:,1]

df = pd.read_csv('scanData.txt',delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]


cartesian = [(r*math.cos(phi*math.pi/180), r*math.sin(phi*math.pi/180)) for r, phi in zip(distance, angle)]
#print(x,y)
x, y = map(list, zip(*cartesian))
print(cartesian)
lr = linear_model.LinearRegression()
lr.fit(x, y)

# Predict data of estimated models
#line_X = np.arange(x.min(), x.max())[:, np.newaxis]
#line_y = lr.predict(line_X)

plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')