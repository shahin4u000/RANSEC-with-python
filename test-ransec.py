import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestClassifier


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

clf = RandomForestClassifier(random_state=0)
X = [[ 1,  2,  3],[11, 12, 13]]
y = [0, 1]
print(clf.fit(X, y))