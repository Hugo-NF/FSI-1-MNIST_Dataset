import cv2 as cv
import numpy as np

class Features:
    "This class is intended to extract features from images"

    def __init__(self, array, rows=28, cols=28, t=30):
        self.array = array
        self.img = array.reshape(rows, cols)
        self.thresh = cv.threshold(self.img, t, 255, 0)
        self.contours, self.hierarchy = cv.findContours(self.thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.cnt = self.contours[0]
        self.perimeter = cv.arcLength(self.cnt, True)
        self.area = cv.contourArea(self.cnt)
        self.hull = cv.convexHull(self.cnt)

    def centroid(self):
        M = cv.moments(self.thresh)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return [cX, cY]

    def axis_least_inertia(self):
        line = cv.fitLine(self.cnt, cv.DIST_L2, 0, 0.01, 0.01)
        return list(map(lambda x:x*100, line))


    def eccentricity(self):
        x, y, w, h = cv.boundingRect(self.cnt)
        return w/h


    def circularity_ratio(self):
        radius = self.perimeter/(2 * np.pi)

        return self.area/((2 * np.pi)*radius)

    def rectangularity(self):
        x, y, w, h = cv.boundingRect(self.cnt)
        rec_area = w * h
        return self.area/rec_area

    def convexity(self):
        hull_perimeter = cv.arcLength(self.hull, True)

        return hull_perimeter/self.perimeter

    def solidity(self):
        hull_area = cv.contourArea(self.hull)
        return self.area/hull_area

