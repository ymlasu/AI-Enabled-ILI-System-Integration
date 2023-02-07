import numpy as np
import cv2
import matplotlib.pyplot as plt 
from scipy.spatial import ConvexHull
import scipy
from mpl_toolkits.mplot3d import Axes3D
import torch


def measure(img_name, seg, var, x, y, z):
    '''
    Measurement function for each image
    Input: 
     One-Hot Encoded Segmentation CxWxH
     Classwise Uncertainty Map CxWxH
     Corresponding X,Y,Z - Maps 
    Returns: 
    Dict of measurements
    {
        corrosion:
            Instance 1: Area, Area Var, Depth Var
            .
            .
            .
            Instance n: Area, Area Var, Depth Var
        crack:
            Instance 1: Length, Length Var, Depth Var
            .
            .
            .
            Instance n: Length, Length Var, Depth Var
        
    }
    '''

    def __area__(cnt):
        '''
        Helper function to measure defect area
        '''
        defect = np.zeros_like(predPitting, dtype = float)
        cv2.drawContours(defect, [cnt], 0, 255, 1)
        pixelIndices = np.where(defect == 255)
        if len(pixelIndices[0]) < 4:
            return None
        xList = x[pixelIndices]
        yList = y[pixelIndices]
        zList = z[pixelIndices]
        pointsList = np.transpose(np.asarray([xList, yList, zList]))
        pointsList = pointsList[~np.all(pointsList == 0, axis = 1)]
        if np.isnan(pointsList).any():
            return None
        try:
            hull = ConvexHull(pointsList)
        except scipy.spatial.qhull.QhullError:
            hull = ConvexHull(pointsList, qhull_options='QJ')
        vertices = hull.vertices.tolist() + [hull.vertices[0]]
        area = hull.area
        return area

    def __depth__(cnt):
        '''
        Helper function to measure defect depth
        Required args: contour instance
        Returns: average depth
        '''
        defect = np.zeros_like(predPitting, dtype = float)
        boundRect = np.zeros_like(predPitting, dtype = float)
        cv2.drawContours(defect, [cnt], 0, 255, -1)
        defectIndices = np.where(defect == 255)
        if len(defectIndices[0]) < 4:
            return None
        xList_defect = x[defectIndices]
        yList_defect = y[defectIndices]
        zList_defect = z[defectIndices]

        boundRectPts = cv2.boundingRect(cnt)
        cv2.rectangle(boundRect, (boundRectPts[0], boundRectPts[1]), (boundRectPts[0] + boundRectPts[2], boundRectPts[1] + boundRectPts[3]), 255, -1)
        boundRectIndices = np.where(boundRect == 255)
        
        # NOTE: This method assumes a small defect size.
        # TODO: Improve upon this method to take into account curvature.
        # depth = np.mean(z[defectIndices]) - np.mean(z[boundRectIndices])
        depth = np.max(z[defectIndices])- np.min(z[boundRectIndices])
        return depth

    def __length__(cnt):
        '''
        Helper function to measure defect length
        '''
        rectImg = np.zeros_like(predCrack, dtype=float)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box[:,1] = np.clip(box[:,1],0,479)
        box[:,0] = np.clip(box[:,0],0,647)
        x_1 = x[box[0][1], box[0][0]]
        y_1 = y[box[0][1], box[0][0]]
        z_1 = z[box[0][1], box[0][0]]
        x_2 = x[box[2][1], box[2][0]]
        y_2 = y[box[2][1], box[2][0]]
        z_2 = z[box[2][1], box[2][0]]
        return scipy.spatial.distance.euclidean([x_1, y_1, z_1], [x_2, y_2, z_2])


    measurements = {'img_index' : img_name, 'corrosion': {}, 'crack': {}}
    
    # Resize seg back to original dims
    seg = cv2.resize(seg[0], (648,480), interpolation=cv2.INTER_NEAREST)

    predPitting = np.ma.masked_where(seg == 1, seg).mask.astype(int)
    predCrack = np.ma.masked_where(seg == 2, seg).mask.astype(int)
    # TODO: Thresholding step for defects (Morph ops?)
    
    # Pitting Defects
    if predPitting.shape != ():
        contours, hierarchy = cv2.findContours(predPitting, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        contoursSorted = sorted(contours, key = lambda x: cv2.arcLength(x, True))
        for i, cnt in enumerate(contoursSorted):
            area = __area__(cnt)
            depth = __depth__(cnt)
            measurements['corrosion'][i] = [area, depth] # TODO: uncertainty
        
    # Cracking defects
    if predCrack.shape != ():
        contours, hierarchy = cv2.findContours(predCrack, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
        contoursSorted = sorted(contours, key = lambda x: cv2.arcLength(x, True))
        for i, cnt in enumerate(contoursSorted):
            # Length of the crack is approximated using the bounding rectangle of the contour
            length = __length__(cnt)
            depth = __depth__(cnt)
            measurements['crack'][i] = [length, depth]
    return measurements

def asme_b31g():
    return "NotImplemented"

def ng_18():
    return "NotImplemented"