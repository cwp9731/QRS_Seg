# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :dl_results
# @Time      :2024/8/15 23:39
# @Author    :Chen
"""
import os
import pickle
import numpy as np
from tqdm import tqdm


def get_points(predict):
    point_list = []
    start = -1
    end = -1
    for i in range(len(predict)):
        if (i == 0) and (predict[i] == 1):
            start = i
        elif (predict[i] == 1) and (predict[i-1] == 0):
            start = i
        elif (i == len(predict)-1) and (predict[i] == 1):
            end = i
        elif (predict[i] == 1) and (predict[i+1] == 0):
            end = i
        if (start != -1) and (end != -1):
            point_list.append([int(start), int(end)])
            start = -1
            end = -1
    return point_list

def Rule1(pred):
    point_pred = get_points(pred)
    for point in point_pred:
        if point[1] - point[0] < 30:
            pred[point[0]:point[1]+1] = 0
    return [pred]


def evaluation_single(point_label, point_pred, tol):
    QRS_start = 0
    QRS_start_find = False
    QRS_end = 0
    QRS_end_find = False
    QRS_start_and_end = 0
    QRS_start_and_end_find = False
    QRS_total = 0
    start_err_list = []
    end_err_list = []
    for label in point_label:
        for pred in point_pred:
            start_err = label[0] - pred[0]
            end_err = label[1] - pred[1]
            if abs(start_err) <= tol/2:
                QRS_start_find = True
                start_err_list.append(start_err)
            if abs(end_err) <= tol/2:
                QRS_end_find = True
                end_err_list.append(end_err)
            if (abs(start_err) <= tol/2) and (abs(end_err) <=tol/2):
                QRS_start_and_end_find = True
        if QRS_start_find:
            QRS_start += 1
            QRS_start_find = False
        if QRS_end_find:
            QRS_end += 1
            QRS_end_find = False
        if QRS_start_and_end_find:
            QRS_start_and_end += 1
            QRS_start_and_end_find = False
        QRS_total += 1
    QRS_err = 0
    for pred in point_pred:
        if pred[1] < point_label[0][0] or pred[0] > point_label[-1][-1]:
            continue
        QRS_err_find = True
        for label in point_label:
            start_err = label[0] - pred[0]
            end_err = label[1] - pred[1]
            if (abs(start_err) <= tol/2) and (abs(end_err) <= tol/2):
                QRS_err_find = False
                continue
        if QRS_err_find:
            QRS_err += 1
    return [QRS_start, QRS_end, QRS_total, QRS_start_and_end, QRS_err, start_err_list, end_err_list]




def evaluation(label, pred):
    tol = 40
    lead_list = list(label.keys())
    for lead in lead_list:
        point_label = label[lead]
        point_pred = get_points(pred[lead][0])
        Result = evaluation_single(point_label, point_pred, tol)
        ON = Result[0]/ Result[2]
        OFF = Result[1]/ Result[2]
        TPR = Result[3] / Result[2]
        FDR = Result[4] / (Result[4]+Result[3])
        return ON, OFF, TPR, FDR


def panduan(point, delet_point):
    # 将数组B转换为集合，元素为元组
    B_set = set(map(tuple, delet_point))

    # 检查数组A中的每个点是否在集合B中
    result = tuple(point) in B_set
    return result


def Rule2(pred_dict):
    tol = 40
    leads = list(pred_dict.keys())
    refer_label = np.zeros_like(pred_dict[leads[0]])[0]

    for lead in leads:
        for i in range(len((pred_dict[lead][0]))):
            if pred_dict[lead][0][i] == 1:
                refer_label[i] = 1
    refer_points = get_points(refer_label)

    count_list = []
    lead_points = [get_points(pred_dict[lead][0]) for lead in leads]
    for index, refer_point in enumerate(refer_points):
        count_list.append(0)
        for lead in range(len(leads)):
            for lead_point in lead_points[lead]:
                if (lead_point[0] >= refer_point[0]) and (lead_point[1] <= refer_point[1]):
                    count_list[index] += 1
                    break
    detel_leads = []
    for lead in range(len(leads)):
        detel_leads.append([])
    for index, refer_point in enumerate(refer_points):
        if count_list[index] < 6:
            for lead in range(len(leads)):
                for lead_point in lead_points[lead]:
                    if (lead_point[0] >= refer_point[0]) and (lead_point[1] <= refer_point[1]):
                        detel_leads[lead].append([lead_point[0], lead_point[1]])
    correction_label = np.zeros((12, len(refer_label)))
    for i, lead in enumerate(leads):
        for point in lead_points[i]:
            if not panduan(point, detel_leads[i]):
                correction_label[i][int(point[0]):int(point[1])+1] = 1
        pred_dict[lead] = [correction_label[i]]
    return pred_dict


def obtain_refer_QRS(local):
    lelf = local[:, 0]
    right = local[:, 1]
    start = np.median(lelf[lelf != -1])
    end = np.median(right[right != -1])
    return int(start), int(end)

def Rule3(pred_dict):
    tol = 40
    pid_list = list(pred_dict.keys())
    leads = list(pred_dict.keys())
    refer_label = np.zeros_like(pred_dict[leads[0]])[0]
    for lead in leads:
        for i in range(len((pred_dict[lead][0]))):
            if pred_dict[lead][0][i] == 1:
                refer_label[i] = 1
    refer_points = get_points(refer_label)
    label_lead_points = np.zeros((12, len(refer_points), 2))
    count_list = []
    lead_points = [get_points(pred_dict[lead][0]) for lead in leads]
    for index, refer_point in enumerate(refer_points):
        count_list.append(0)
        for lead in range(len(leads)):
            is_null = True
            for lead_point in lead_points[lead]:
                if (lead_point[0] >= refer_point[0]) and (lead_point[1] <= refer_point[1]):
                    count_list[index] += 1
                    label_lead_points[lead][index] = [lead_point[0], lead_point[1]]
                    is_null = False
                    continue
            if is_null:
                label_lead_points[lead][index] = [-1, -1]
    t = 0
    for index, refer_point in enumerate(refer_points):
        if count_list[index] >= 6:
            start, end = obtain_refer_QRS(label_lead_points[:, index-t, :])
            for lead in range(len(label_lead_points)):
                if label_lead_points[lead][index-t][0] == -1:
                    label_lead_points[lead][index-t] = [start, end]
                if abs(label_lead_points[lead][index-t][0] - start) > tol/4:
                    label_lead_points[lead][index-t][0] = start
                if abs(label_lead_points[lead][index-t][1] - end) >tol/4:
                    label_lead_points[lead][index-t][1] = end
    correction_label = np.zeros((12, len(refer_label)))
    for i, lead in enumerate(leads):
        for point in label_lead_points[i]:
            correction_label[i][int(point[0]):int(point[1])+1] = 1
        pred_dict[lead] = [correction_label[i]]
    return pred_dict

def generate_results(data_list):
    post_index = ['unet', 'unet+s1',  'unet+s2',  'unet+s3', 'unet+s1+s2', 'unet+s1+s3', 'unet+s2+s3', 'unet+s1+s2+s3']
    for data in data_list:
        for post in post_index:
            with open(os.path.join('./result/', data + '.pkl'), 'rb') as input:
                (label_dict, pred_dict) = pickle.load(input)
            if 's1' in post:
                pred_dict = Rule1(pred_dict)
            if 's2' in post:
                pred_dict = Rule2(pred_dict)
            if 's3' in post:
                pred_dict = Rule3(pred_dict)
            evaluation(label_dict, pred_dict)


