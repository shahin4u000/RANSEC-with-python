import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import ransac, LineModelND, CircleModel
import math
import cv2

df = pd.read_csv(r'scan-data\capture1.csv',delimiter=',')

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

############# RANSAC Start ##################
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
    elif detectedByRansac.size >=25:
        inliersArray = np.concatenate((inliersArray,detectedByRansac))
        #print('inliersArray: ', inliersArray)
    #update the data with outliers and remove inliers
    
    
    data = np.column_stack([data[outliers, 0],data[outliers, 1]])
    #print("inliers: ", inliers)
    #print("wihtout: ", data)
    dataSize = data.size

    ## If you want to see how the RANSAC works uncomment the follwoing code

    ''' fig, ax = plt.subplots()
    ####
    #### test
    ax.plot(data[:, 0], data[:, 1], '.r', alpha=0.6,
            label='Outlier data')
    ax.plot(inliersArray[:, 0], inliersArray[:, 1], '.b', alpha=0.6,
            label='Inlier data')
    ax.legend(loc='top left')
    plt.show()
    plt.pause(0.0001)  '''



########### DBSCAN start ##################

X = StandardScaler().fit_transform(inliersArray)
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_  


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)



print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


# Black removed and is used for noise instead.
finalData= np.array([])
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)
    print('class_member_mask: ', class_member_mask)
    xy = inliersArray[class_member_mask & core_samples_mask]
    detectedByDBSCAN= np.column_stack([xy[:, 0], xy[:, 1]])
    
    if finalData.size == 0:
        finalData = detectedByDBSCAN
    else:
        finalData = np.concatenate((finalData,detectedByDBSCAN))
        print('finalData: ', finalData)
    #update the data with outliers and remove inliers
    
   

    #xy = data[class_member_mask & core_samples_mask]
plt.plot(finalData[:, 0], finalData[:, 1], 'o', markeredgecolor='k', markersize=5)
plt.show()
print("i am size: ",xy)

################
Thetas = []
Ranges = []
maxRange = 0

finalData = [i for i in data1 if i in finalData]

for row in finalData:
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
    cv2.drawContours(im, [triangleCnt.astype(int)], 0, 255, -1)
cv2.imwrite('result.png', im)

#plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.show()