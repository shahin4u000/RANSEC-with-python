#!/usr/bin/env python3
# Skinny Triangle Formula: https://en.wikipedia.org/wiki/Skinny_triangle

import math
import cv2
import csv, sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # Pick up name of file to analyze
    #if len(sys.argv) != 2:
        #print('Usage: analyse data.csv')
        #sys.exit(1)

    # Load values, expected format is:
    # theta,range
    # where theta is in degrees and range is in mm
    Thetas = []
    Ranges = []
    maxRange = 0
    filename = r'C:\Users\kgoni\Desktop\RANSEC-with-python\current\scan-data-for-Room-1.csv'
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            theta = float(row[0])
            r = float(row[1])
            #print(f'DEBUG: theta: {theta}, range: {r}')
            if r> maxRange:
                maxRange = r
            Ranges.append(r)
            Thetas.append(theta)

    # Plot what we found - i.e. a triangle centred on lidar stretching out to 2 pts on  walls
    # Work out size of canvs
    maxRange = int(maxRange+1)

    h = w = 2 * maxRange
    cx = cy = maxRange
    print(maxRange)
    # Make blank canvas
    im = np.zeros((h,w), np.uint8)
    # Centre point, in every triangle
    pt1 = (cy, cx)
    print('center',pt1)

    # Join up last point with first to complete the circle
    Ranges.append(Ranges[0])
    Thetas.append(Thetas[0])
    print("theta: ",Thetas)

    print("Ranges: ",Ranges)
    for i in range(1,len(Ranges)):
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
        print(triangleCnt)
        cv2.drawContours(im, [triangleCnt.astype(int)], 0, 255, 255)
    image = im.copy()

#    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 #   thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
  #  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
   # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    minLineLength = 7500
    maxLineGap = 58
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), (36, 255, 12), 3)

#    cv2.imwrite('thresh', thresh)
 #   cv2.imwrite('close', close)
    cv2.imwrite('image', image)
    cv2.imwrite('result1.jpg', im)
    print(im)
