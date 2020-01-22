import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.measure import ransac, LineModelND, CircleModel
import math
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


df = pd.read_csv(r'C:\Users\kgoni\Desktop\RANSEC-with-python\scan-data\capture2.csv',delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]

x= angle
y= distance

cartesian = [(r*math.sin(phi*math.pi/180),r*math.cos(phi*math.pi/180) ) for r, phi in zip(distance, angle)]
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
#print ("inliersArray", inliersArray)
dataSize = data.size
#print("data size", dataSize)

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
    line_y_robust = model_robust.predict_y(line_x)
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
    #print("inliers: ", inliers)
    #print("wihtout: ", data)
    dataSize = data.size
    fig, ax = plt.subplots()
    ####
    #### test
    ax.plot(data[:, 0], data[:, 1], '.r', alpha=0.6,
            label='Outlier data')
    ax.plot(inliersArray[:, 0], inliersArray[:, 1], '.b', alpha=0.6,
            label='Inlier data')
    ax.legend(loc='top left')
    plt.show()
    plt.pause(0.0001) 


centers = inliersArray

X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result


# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=5)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()