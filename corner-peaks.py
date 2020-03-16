import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import ransac, LineModelND, CircleModel
from skimage.feature import peak_local_max, corner_peaks, corner_shi_tomasi
import math

df = pd.read_csv(r'C:\Users\kgoni\Desktop\RANSEC-with-python\scan-data\capture1.csv',delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]

x= angle
y= distance

cartesian = [( r*math.sin(phi*math.pi/180),r*math.cos(phi*math.pi/180)) for r, phi in zip(distance, angle)]
#print(x,y)
x, y = map(list, zip(*cartesian))
#print(cartesian)

# coverting this into 2d array
x=  np.array(x)
y=  np.array(y)

x=x.reshape(-1, 1)
y=y.reshape(-1, 1)


data = np.column_stack([x, y])
data1 = data

inliersArray = np.array([])
print ("inliersArray", inliersArray)
dataSize = data.size
print("data size", dataSize)

while dataSize >=20:    
    print('dataSize: ', dataSize)
    model = LineModelND()
    model.estimate(data)
    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                            residual_threshold=50, max_trials=10000)
    outliers = inliers == False
    # generate coordinates of estimated models
    line_x = np.arange(x.min(), x.max()) #[:, np.newaxis]
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_y)
    detectedByRansac= np.column_stack([data[inliers, 0],data[inliers, 1]])
    print('detectedByRansac: ', detectedByRansac)
    
    #store the inliers into inliers array
    if inliersArray.size == 0:            
        inliersArray = detectedByRansac
        print('inliersArray: ', inliersArray)
    else :
        if detectedByRansac.size >=30:
            inliersArray = np.concatenate((inliersArray,detectedByRansac))
            print('inliersArray: ', inliersArray)
    #update the data with outliers and remove inliers
    
    
    data = np.column_stack([data[outliers, 0],data[outliers, 1]])
    print("inliers: ", inliers)
    print("wihtout: ", data)
    dataSize = data.size
    fig, ax = plt.subplots()
    ####
    #### test
    ax.plot(data[:, 0], data[:, 1], '.r', alpha=0.6,
            label='Outlier data')
    ax.plot(inliersArray[:, 0], inliersArray[:, 1], '.b', alpha=0.6,
            label='Inlier data')
    ax.legend(loc='top left')
    '''plt.show()
    plt.pause(0.0001)'''  

print("hi corner; ",corner_peaks(corner_shi_tomasi(inliersArray), min_distance=1))
fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], '.r', alpha=0.6,
        label='Outlier data')
ax.plot(inliersArray[:, 0], inliersArray[:, 1], '.b', alpha=0.6,
        label='Inlier data')

ax.legend(loc='top left')
plt.show()