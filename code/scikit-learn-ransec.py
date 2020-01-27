import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import linear_model, datasets
import math
from sklearn.linear_model import RANSACRegressor, LinearRegression

# getting data from a text file
df = pd.read_csv(r"/home/kazi/Desktop/RANSEC-with-python/code/capture.csv",delimiter=',')

angle = df.values[:,0]
distance = df.values[:,1]
cartesian = [(r*math.cos(phi*math.pi/180), r*math.sin(phi*math.pi/180)) for r, phi in zip(distance, angle)]
x, y = map(list, zip(*cartesian))
#print(cartesian)


# coverting this into 2d array
x=  np.array(np.floor(x))
y=  np.array(np.floor(y))
#print(x,y)
a= x
b= y

x_shape = int(np.max(a) - np.min(a))
y_shape = int(np.max(b) - np.min(b))

#im = np.zeros((x_shape+1, y_shape+1))

#indices = np.stack([a-1,b-1], axis =1).astype(int)
#im[indices[:,0], indices[:,1]] = 1

#plt.imshow(im)

X=x.reshape(-1, 1)
Y=y.reshape(-1, 1)

#lr = linear_model.LinearRegression()
#lr.fit( y,x)

#from sklearn.linear_model import LinearRegression, RANSACRegressor

ransac = RANSACRegressor(
                         max_trials=15000,
                         min_samples=200,
                         #residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                         residual_threshold=1,
                         )
ransac.fit(X,Y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(-5,5)

line_y_ransac = ransac.predict(line_X[:, np.newaxis])
print(X[inlier_mask])
print(y[inlier_mask])
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='.', label='Outliers')

plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='.', label='Inliers')
plt.plot(line_X, line_y_ransac, color='red')
<<<<<<<

=======
#plt.xlabel('Average number of rooms [RM]')
#plt.ylabel('Price in $1000\'s [MEDV]')
#plt.legend(loc='upper left')
>>>>>>>


plt.tight_layout()
plt.show()

########################
