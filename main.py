# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :main
# @Time      :2024/8/15 23:34
# @Author    :Chen
"""
import numpy as np
import random
import os
import torch
import pickle
from sklearn.model_selection import KFold
from dl_data import my_dataset
from dl_pipeline import train_dl, test_dl
from torch.utils.data import DataLoader
from dl_results import generate_results


def setup_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)



def train_models(data_name):
    data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(data_path, 'data', data_name + '.pkl'), 'rb') as input:
        Dataset = pickle.load(input)

    batch_size = 32

    record_list = np.array(list(Dataset[0].keys()))
    kf = KFold(n_splits=5, shuffle=True)
    fold_idx = kf.split(record_list)
    fold_idx = [[train_idx, test_idx] for train_idx, test_idx in fold_idx]
    pred_dict = dict()


    time = 1
    for train_idx, test_idx in fold_idx:
        print('5 fold-No.{}'.format(time))
        train_data = DataLoader(my_dataset(Dataset, record_list[train_idx]), batch_size=batch_size, shuffle=True, drop_last=False)
        test_data = DataLoader(my_dataset(Dataset, record_list[test_idx]), batch_size=32, shuffle=False, drop_last=False)

        train_dl(train_data, test_data, time,  data_name)
        pred_dict = test_dl(test_data, time, data_name, pred_dict)

        time += 1


    file_path = r"./result"
    if not os.path.exists(file_path):
        # 如果路径不存在，则创建该路径
        os.makedirs(file_path)
    file_path = os.path.join(file_path, data_name+".pkl")
    with open(file_path, 'wb') as output:  # Overwrites any existing file.
        pickle.dump((Dataset[1], pred_dict), output)



def main():
    setup_seed()
    data_list = ["ludb", "real_data"]
    for data_name in data_list:
        train_models(data_name)

    generate_results(data_list)


if __name__ == "__main__":
    main()
