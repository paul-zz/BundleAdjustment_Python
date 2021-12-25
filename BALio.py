# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 19:07:35 2021

@author: Zijian Zhong

This script is to read data from BAL dataset
"""

import numpy as np
import numpy.linalg as li
from re import split as resplit
import Transforms

def read_bal(filename):
    
    """
    Read BAL dataset from txt file
    input:
        filename: txt filename
    output: 
        camera_index, point_index, cameras, points, observations
    """
    
    camera_index = [] # The camera index in each observation
    point_index = [] # The point index in each observation
    cameras = [] # Parameters of each camera (R(3x1 Rodrigues), t(3x1 Vector), f, k1, k2)
    points = []  # 3D Coordinates of each point (x, y, z)
    observations = [] # 2D Coordinates of each pixel (x, y) 
    with open(filename, 'r') as f:
        dataset_params = f.readline().split(' ')
        num_cams = int(dataset_params[0])
        num_points = int(dataset_params[1])
        num_observations = int(dataset_params[2])
        
        for i in range(num_observations):
            line = f.readline()
            line_split = resplit('[ ]+', line)
            camera_index.append(int(line_split[0]))
            point_index.append(int(line_split[1]))
            observations.append([float(line_split[2]), float(line_split[3])])
        
        for i in range(num_cams):
            cam_param = []
            for j in range(9):
                line = f.readline()
                cam_param.append(float(line))
            cameras.append(cam_param)
        
        for i in range(num_points):
            pt_param = []
            for j in range(3):
                line = f.readline()
                pt_param.append(float(line))
            points.append(pt_param)  
            
    return camera_index, point_index, cameras, points, observations

def create_problem(filename):
    camera_index, point_index, cameras, points, observations = read_bal(filename)
    cameras = np.array(cameras)
    points = np.array(points)
    cameras, points = normalize(cameras, points)
    return camera_index, point_index, cameras, points, observations
    
def camera_to_rod_center(camera):
    rotation_ref = camera[0:3]
    inverse_rotation = - rotation_ref
    center = Transforms.rotatePoint(inverse_rotation, camera[3:6])
    center *= -1.0
    return rotation_ref, center

def rod_center_to_camera(rodrigues, center):
    center = Transforms.rotatePoint(rodrigues, center)
    camera = np.zeros(6)
    camera[0:3] = rodrigues
    camera[3:6] = center
    return camera

def normalize(cameras, points):
    num_points = len(points)
    median = np.median(points, axis=0)
    submedian = li.norm(points - np.tile(median, (num_points, 1)),1, axis=0)
    median_absolute_deviation = np.median(submedian)
    scale = 100.0 / median_absolute_deviation
    
    points = scale * (points - np.tile(median, (num_points, 1)))
    for i in range(len(cameras)):
        camera = cameras[i]
        rodrigues, center = camera_to_rod_center(camera)
        center = scale * (center - median)
        # print(rod_center_to_camera(rodrigues, center))
        cameras[i,0:6] = rod_center_to_camera(rodrigues, center)
    return cameras, points