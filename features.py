import cv2 as cv
import numpy as np

class Features:
    "This class is intended to extract features from images"

    def __init__(self, array, rows, cols, t):
        self.array = array
        self.img = array.reshape(rows, cols)
        self.thresh = cv.threshold(self.img, t, 255, 0)
        self.contours, self.hierarchy = cv.findContours(self.thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.cnt = self.contours[0]

    def centroid(self):
        M = cv.moments(self.thresh)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return [cX, cY]

    def ALI(self):
        line = cv.fitLine(self.cnt, cv.DIST_L2, 0, 0.01, 0.01)
        return list(map(lambda x:x*100, line))

    def elongation(self):
        x, y, w, h = cv.boundingRect(self.cnt)
        return 1 - w/h

    def eccentricity(self):
        x, y, w, h = cv.boundingRect(self.cnt)
        return w/h
