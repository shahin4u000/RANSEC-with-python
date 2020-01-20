import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import ransac, LineModelND, CircleModel
from skimage.feature import peak_local_max, corner_peaks, corner_shi_tomasi
import math
import cv2

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
    #print('dataSize: ', dataSize)
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
    #print('detectedByRansac: ', detectedByRansac)
    
    #store the inliers into inliers array
    if inliersArray.size == 0:            
        inliersArray = detectedByRansac
        #print('inliersArray: ', inliersArray)
    elif detectedByRansac.size >=30:
        inliersArray = np.concatenate((inliersArray,detectedByRansac))
        #print('inliersArray: ', inliersArray)
    #update the data with outliers and remove inliers
    
    
    data = np.column_stack([data[outliers, 0],data[outliers, 1]])
    #print("inliers: ", inliers)
    #print("wihtout: ", data)
    dataSize = data.size



################
print("hi :", inliersArray.size)
Thetas = []
Ranges = []
maxRange = 0

""" r = np.sqrt(inliersArray[0]**2+inliersArray[1]**2)
t = np.arctan2(inliersArray[1],inliersArray[0])
print("r: ", r)
print("t:", t)
csv_reader = zip(r,t) """

print("i am in polar: ",inliersArray)
for row in inliersArray:
    #print("row ",row)
    r = row[0]
    theta = row[1]
    
    #print(f'DEBUG: theta: {theta}, range: {r}')
    if r> maxRange:
        maxRange = r
    if theta > maxRange:
        maxRange = theta
    Ranges.append(r)
    Thetas.append(theta)

print("maxrange : ",maxRange)
# Plot what we found - i.e. a triangle centred on lidar stretching out to 2 pts on  walls
# Work out size of canvs
maxRange = int(maxRange+1)
h = w = 2 * maxRange
cx = cy = maxRange
# Make blank canvas
im = np.zeros((h,w), np.uint8)
# Centre point, in every triangle
pt1 = (cy, cx)
# Join up last point with first to complete the circle
Ranges.append(Ranges[0])
Thetas.append(Thetas[0])
for i in range(1,len(Thetas)):
    r = Ranges[i-1]
    theta = Thetas[i-1]
    x = r
    y = theta
    pt2 = (cx + x, cy - y)
    r = Ranges[i]
    theta = Thetas[i]
    x = r
    y = theta
    pt3 = (cx + x, cy - y)
    triangleCnt = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(im, [triangleCnt.astype(int)], 0, 255, 0)
cv2.imwrite('result.png', im)


fig, ax = plt.subplots()
ax.plot(data[:, 0], data[:, 1], '.r', alpha=0.6,
        label='Outlier data')
ax.plot(inliersArray[:, 0], inliersArray[:, 1], '.b', alpha=0.6,
        label='Inlier data')

ax.legend(loc='top left')
plt.show()