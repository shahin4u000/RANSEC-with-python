import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# from skimage.measure import LineModelND, ransac
from sklearn import linear_model, datasets
from skimage.measure import ransac, LineModelND

import math



df = pd.read_csv('scanData.txt',delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]


cartesian = [(r*math.cos(phi*math.pi/180), r*math.sin(phi*math.pi/180)) for r, phi in zip(distance, angle)]
#print(x,y)
x, y = map(list, zip(*cartesian))
#print(cartesian)
#plt.scatter(y,x)
#plt.show()


# generate coordinates of line
#x = np.arange(-200, 200)

#y = 0.2 * x + 20
data = np.column_stack([x, y])

print(data)
#print("x: ", x ,  " y: ", y)
# add gaussian noise to coordinates
#noise = np.random.normal(size=data.shape)
#data += 0.5 * noise
#data[::2] += 5 * noise[::2]
#data[::4] += 20 * noise[::4]

# add faulty data
#faulty = np.array(30 * [(180., -100)])
#faulty += 10 * np.random.normal(size=faulty.shape)
#data[:faulty.shape[0]] = faulty

# fit line using all data
model = LineModelND()
model.estimate(data)    # estimate random data

# robustly fit line only using inlier data with RANSAC algorithm
model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                               residual_threshold=10, max_trials=1000)
outliers = inliers == False

# generate coordinates of estimated models
line_x = np.arange(-250, 250)
line_y = model.predict_y(line_x)
line_y_robust = model_robust.predict_y(line_x)
print(line_y_robust)

fig, ax = plt.subplots()
ax.plot(data[inliers, 0], data[inliers, 1], '.b', alpha=0.6,
        label='Inlier data')
ax.plot(data[outliers, 0], data[outliers, 1], '.r', alpha=0.6,
        label='Outlier data')
ax.plot(line_x, line_y, '-r', label='Line model from all data')
ax.plot(line_x, line_y_robust, '-b', label='Robust line model')
ax.legend(loc='lower left')
plt.show()