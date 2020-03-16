import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import ransac, LineModelND, CircleModel
import math  


df = pd.read_csv('xy.csv',delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]

x= angle
y= distance


# coverting this into 2d array
x=  np.array(x)
y=  np.array(y)

x=x.reshape(-1, 1)
y=y.reshape(-1, 1)

data = np.column_stack([x, y])

model = LineModelND()
model.estimate(data)
# robustly fit line only using inlier data with RANSAC algorithm
model_robust, inliers = ransac(data, LineModelND, min_samples=3,
                               residual_threshold=2, max_trials=100)
outliers = inliers == False

# generate coordinates of estimated models
line_x = np.arange(x.min(),x.max())  #[:, np.newaxis]
line_y = model.predict_y(line_x)
line_y_robust = model_robust.predict_y(line_x)

fig, ax = plt.subplots()
ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
        label='Outlier data')
ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
        label='Inlier data')
print("data: ", data)
print(data[inliers, 0], data[inliers, 1])
#ax.plot(line_x, line_y, '-k', label='Line model from all data')
#ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
#ax.legend(loc='lower left')
plt.show()