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
import time
import os
import shutil


# =============================================================================
# 定义超参数 
# =============================================================================
Distance_threshold = 1e-4  # 迭代类中心收敛判定
Cluster_threshold = 1e-4  # 类中心间merge判定

# =============================================================================
# 定义函数
# =============================================================================
def euclidean_dist(a, b):
    '''
    计算两点的距离(L2范数)
    '''
    return np.linalg.norm(np.array(a) - np.array(b), ord=2)


def shift_blob(Center_point, Whole_points, Sigma):
    '''
    完成公式(18)的迭代，由X(n)得到X(n+1)
    '''
    center_point_new = np.zeros(Center_point.shape)
    data_diff = np.zeros((Whole_points.shape[0], Center_point.shape[0]))
    for i in range(Whole_points.shape[0]):
        for j in range(Center_point.shape[0]):
            data_diff[i, j] = euclidean_dist(Whole_points[i, :], Center_point[j, :])
    data_gussion_distance = np.exp((data_diff) ** 2 / (2 * Sigma ** 2) * (-1))
    for center_label in range(Center_point.shape[0]):
        for tick_label in range(Center_point.shape[1]):
            center_point_new[center_label, tick_label] = np.sum(
                Whole_points[:, tick_label] * data_gussion_distance[:, center_label]) / np.sum(
                data_gussion_distance[:, center_label])
    return center_point_new


def need_shifting_flag(center_point_new, center_point):
    '''
    判断|X(n+1)-X(n)|<epsilon,是否继续迭代
    '''
    distance_max = 0.0
    for center_label in range(center_point_new.shape[0]):
        dis = euclidean_dist(center_point_new[center_label, :], center_point[center_label, :])
        if dis > Distance_threshold:
            distance_max = max(distance_max, dis)
    if (distance_max < Distance_threshold):
        iteration_flag = False
    else:
        iteration_flag = True
    return iteration_flag


def iterate_blob(center_point_new, whole_points, Gaussian_sigma):
    '''
    判断收敛时跳出循环,center_point_final为最终效果,iteration_time为迭代次数 
    '''
    iteration_time = 1
    while True:
        center_point_final = shift_blob(center_point_new, whole_points, Gaussian_sigma)
        if need_shifting_flag(center_point_final, center_point_new):
            center_point_new = center_point_final
            iteration_time += 1
        else:
            break
    return center_point_final, iteration_time


def Merge_list(a):
    '''
    列表元素合并&去重
    '''
    list_merge = []
    for sublist in a:
        list_merge.extend(sublist)
    return list(set(list_merge))


def Blob_merge(center_point_final):
    '''
    将输入的K个类中心根据|x_2-x_1|<epsilon确定是否需要合并
    '''
    merge_blob_index = []  # 记录两两距离接近的点
    merge_index_plus = []  # 记录需要merge的一簇blob center
    # 计算两两距离接近的blob center 
    for i in range(center_point_final.shape[0]):
        for j in range(center_point_final.shape[0]):
            if (i < j) and euclidean_dist(center_point_final[i, :], center_point_final[j, :]) < Cluster_threshold:
                merge_blob_index.append([i, j])
    # 记录需要merge的一簇blob center 
    for element in merge_blob_index:
        if (element[0] not in Merge_list(merge_index_plus)) and (element[1] not in Merge_list(merge_index_plus)):
            merge_index_plus.append(element.copy())
        else:
            for element2 in merge_index_plus:
                if (element[0] in element2) and (element[1] not in element2):
                    element2.append(element[1])
                elif (element[0] not in element2) and (element[1] in element2):
                    element2.append(element[0])
                    # 删除接近的点，添加融合的点
    new_blob_array = np.zeros((len(merge_index_plus), 2))
    for merge_blob in range(len(merge_index_plus)):
        new_blob_array[merge_blob, :] = np.mean(center_point_final[merge_index_plus[merge_blob], :], axis=0)
    center_point_final_merge1 = np.delete(center_point_final, Merge_list(merge_index_plus), axis=0)  # 删除merge后的点
    center_point_final_merge = np.concatenate((center_point_final_merge1, new_blob_array), axis=0)  # 添加merge后的新点
    return center_point_final_merge, merge_index_plus

def main_SSF(Data_general,Gaussian_sigma_set):
    start = time.time()
    Gaussian_sigma = Gaussian_sigma_set
    os.mkdir('blob_center')
    clusterCenterList=[]

    whole_points = Data_general
    center_point_init = np.array(whole_points)
    
    # 第一次X(0)=x_0,所有数据点均作为类中心
    center_point_new = shift_blob(center_point_init, whole_points, Gaussian_sigma)
    # 迭代找到sigma_1时的收敛点blob center 
    center_point_final, iteration_time = iterate_blob(center_point_new, whole_points, Gaussian_sigma)
    np.save('blob_center\\blob_1', center_point_final)
    # 确认merge后的blob center是否可以继续merge 
    while True:
        center_point_final, merge_index_plus = Blob_merge(center_point_final)
        if len(merge_index_plus) == 0:
            break
    
    # =============================================================================
    # 记录结果
    # =============================================================================
    
    np.save('blob_center\\Data_general', Data_general)
    
    np.save('blob_center\\blob_merge_1', center_point_final)
    iteration_each_step = []
    iteration_each_step.append(iteration_time)
    clusterCenterList.append(center_point_final.shape[0])
    # =============================================================================
    # 正式算法迭代 
    # =============================================================================
    iteration_step = 2

    while True:
        # 计算sigma_k
        Gaussian_sigma *= 1.029
        # 迭代找到sigma_k时的收敛点blob center 
        center_point_final, iteration_time = iterate_blob(center_point_final, whole_points, Gaussian_sigma)
        np.save('blob_center\\blob_' + str(iteration_step), center_point_final)
        iteration_each_step.append(iteration_time)
        # 确认merge后的blob center是否可以继续merge 
        while True:
            center_point_final, merge_index_plus = Blob_merge(center_point_final)
            if len(merge_index_plus) == 0:
                break
        np.save('blob_center\\blob_merge_' + str(iteration_step), center_point_final)
        clusterCenterList.append(center_point_final.shape[0])
        iteration_step += 1
        if center_point_final.shape[0]==1:
            break
        
    # 算lifetime用
    clusterCenterList=np.array(clusterCenterList)
    # 选类中心和sigma 
    clusterCenterIndex=np.where(clusterCenterList==np.argmax(np.bincount(clusterCenterList)))
    chooseSigmaIndex=np.int(np.min(clusterCenterIndex))+1
    chooseSigma=Gaussian_sigma_set*1.029**chooseSigmaIndex
    chooseClusterCenter=np.load('blob_center\\blob_merge_'+str(chooseSigmaIndex)+'.npy')
#     删除blob_center文件夹
    shutil.rmtree(r'blob_center') 
    end = time.time()
    
    
    print('程序迭代次数：%d次' % iteration_step)
    print('程序执行时间：%3f秒' % (end - start))
    return clusterCenterList,chooseSigma,chooseClusterCenter