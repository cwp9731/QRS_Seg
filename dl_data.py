# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :dl_data
# @Time      :2024/8/15 23:49
# @Author    :Chen
"""
import numpy as np
import random
import torch
from torch.utils.data import Dataset

class my_dataset(Dataset):
    def __init__(self, Dataset, idxes):
        self.x, self.y, self.pids = self.get_data(Dataset, idxes)

    def __len__(self):
        return len(self.x)

    def get_data(self, Dataset, idxes):
        (data_dict, label_dict) = Dataset
        leads = list(data_dict[idxes[0]].keys())
        x = []
        y = []
        pids = []
        for idx in idxes:
            for lead in leads:
                signal = data_dict[idx][lead]
                x.append((signal - np.mean(signal)) / np.std(signal))
                label = np.zeros(len(signal))
                for point in label_dict[idx][lead]:
                    label[point[0]:point[1] + 1] = 1
                y.append(label)
                pids.append(idx+'_'+lead)
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        pids = np.array(pids)

        return [x, y, pids]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        pid = self.pids[index]
        if x.dtype != float:
            x = np.array(x, dtype=float)
        x = torch.tensor(x, dtype=torch.float)
        x = x.unsqueeze(0)
        y = torch.tensor(y, dtype=torch.float)
        y = y.unsqueeze(0)
        return x, y, pid


