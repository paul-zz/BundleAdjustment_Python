# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:01:38 2021

@author: Zijian Zhong

Rigid Body Transform Functions and Conversions.
Transforms: 
    translation, rotation, reprojection
Conversions:
    Not yet. XD
"""

import numpy as np
import numpy.linalg as li

def rotatePoint(rodrigues, point):
    # Rotate a point by rodrigues formula
    # if type(rodrigues) != np.ndarray:
    #     rodrigues = np.array(rodrigues, dtype=np.float64)
    # if type(point) != np.ndarray:
    #     point = np.array(point, dtype=np.float64)
    theta2 = np.dot(rodrigues, rodrigues)
    if (theta2 > np.finfo(np.float64).eps):
        theta = np.sqrt(theta2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        theta_inv = 1.0 / theta
        w = theta_inv * rodrigues
        w_cross_pt = np.cross(w, point)
        tmp = np.dot(w, point) * (1 - cos_theta)
        result = point*cos_theta + w_cross_pt*sin_theta + w*tmp
    else:
        w_cross_pt = np.cross(rodrigues, point)
        result = point + w_cross_pt
    return result

def project(camera, point):
    # Project the point into pixel plane
    # if type(camera) != np.ndarray:
    #     camera = np.array(camera, dtype=np.float64)
    # if type(point) != np.ndarray:
    #     point = np.array(point, dtype=np.float64)
    rot = camera[0:3]
    trans = camera[3:6]
    focal = camera[6]
    k1 = camera[7]
    k2 = camera[8]
    # Rotate and translate
    point = rotatePoint(rot, point) + trans
    # Compute distortion center
    xp = -point[0]/point[2]
    yp = -point[1]/point[2]
    
    # Compute distorted pixel point
    r2 = xp**2 + yp**2
    distortion = 1.0 + r2 * (k1 * k2 * r2)
    projection = focal*distortion*np.array([xp, yp])
    return projection

def transform(camera, point):
    rot = camera[0:3]
    trans = camera[3:6]
    # Rotate and translate
    trans_point = rotatePoint(rot, point) + trans
    return trans_point