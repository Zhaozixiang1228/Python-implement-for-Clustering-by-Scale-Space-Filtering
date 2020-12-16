# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:45:42 2020

@author: Zixiang Zhao (zixiangzhao@stu.xjtu.edu.cn)

Python implement for "Clustering by Scale-Space Filtering" (IEEE TPAMI 2000)

https://ieeexplore.ieee.org/document/895974
"""
import sklearn.datasets
import matplotlib.pyplot as plt  
import numpy as np
import imageio
import os
import cv2
from SSF import main_SSF

Data_general, Data_labels = sklearn.datasets.make_blobs(
    n_samples=300,
    n_features=2,
    centers=20,
    cluster_std=0.05,
    random_state=0,
    center_box=(-1, 1))
# sigma初始化的值
Gaussian_sigma_set=0.05
# SSF计算输出lifetime, sigma, 类中心
clusterCenterList,chooseSigma,chooseClusterCenter=main_SSF(Data_general,Gaussian_sigma_set)
