#!/usr/bin/env python3
# Skinny Triangle Formula: https://en.wikipedia.org/wiki/Skinny_triangle

import math
import cv2
import csv, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


Thetas = []
Ranges = []
maxRange = 0
csvfile = r'C:\Users\kgoni\Desktop\RANSEC-with-python\scan-data\capture1.csv'

df  = pd.read_csv(csvfile, delimiter=',')
angle = df.values[:,0]
distance = df.values[:,1]
csv_reader = zip(angle,distance)
print(csv_reader)
for row in csv_reader:
    print("row ",row)
    theta = row[0]
    r = row[1]
    print(f'DEBUG: theta: {theta}, range: {r}')
    if r> maxRange:
        maxRange = r
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
for i in range(1,len(Ranges)):
    print("i: ",i)
    print("ranges: ", len(Ranges))
    r = Ranges[i-1]
    theta = np.deg2rad(Thetas[i-1])
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    pt2 = (cx + x, cy - y)
    r = Ranges[i]
    theta = np.deg2rad(Thetas[i])
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    pt3 = (cx + x, cy - y)
    triangleCnt = np.array( [pt1, pt2, pt3] )
    cv2.drawContours(im, [triangleCnt.astype(int)], 0, 255, 0)
cv2.imwrite('result.png', im)
