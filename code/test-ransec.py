import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import linear_model, datasets
import math

# getting data from a text file
from sklearn.linear_model import RANSACRegressor

df = pd.read_csv('scanData.txt',delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]
cartesian = [(r*math.cos(phi*math.pi/180), r*math.sin(phi*math.pi/180)) for r, phi in zip(distance, angle)]
x, y = map(list, zip(*cartesian))
#print(x)


# coverting this into 2d array
x=  np.array(x)
y=  np.array(y)

x=x.reshape(-1, 1)
y=y.reshape(-1, 1)

lr = linear_model.LinearRegression()
lr.fit(x, y)
ransac = RANSACRegressor(max_trials=1000,min_samples=300)
ransac.fit(x, y)

# Predict data of estimated models
line_X = np.arange(x.min(), x.max())[:, np.newaxis]
print(line_X)
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_y)
print(line_y_ransac)
plt.scatter(x,y, color='yellowgreen', marker='.',
            label='Inliers')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=1,
         label='RANSAC regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()