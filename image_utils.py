'''
Image processing tools and helpers
'''
import numpy as np
import cv2


def detect_blur(img, thresh = 100):
    '''
    Lightweight blur-detection using variance of laplacian
    Input: img
    Returns: blur
    '''
    return bool(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < thresh)