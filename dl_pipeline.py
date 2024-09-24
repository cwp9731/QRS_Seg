# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :dl_pipeline
# @Time      :2024/8/15 23:38
# @Author    :Chen
"""
import os
from dl_models import UNet
import torch
import torch.optim as optim
from tqdm import tqdm
import copy
import numpy as np


def train_single(data, model, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels, pids in data:
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        loss = torch.nn.BCELoss()(out, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item() * inputs.shape[0]
    epoch_loss = running_loss / len(data.dataset)
    return epoch_loss


def val_single(data, model, device):
    model.eval()
    running_loss = 0.0
    for inputs, labels, pids in data:
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        loss = torch.nn.BCELoss()(out, labels)
        running_loss += loss.item() * inputs.shape[0]
    epoch_loss = running_loss / len(data.dataset)
    return epoch_loss


def train_dl(train_data, test_data, time, data_name):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=1e-4, weight_decay=0)

    model_path = os.path.join('models', data_name)
    if not os.path.exists(model_path):
        # 如果路径不存在，则创建该路径
        os.makedirs(model_path)
    model_filepath = os.path.join(model_path, 'model_' + str(time))

    patience = 15
    max_epochs = 400
    best_loss = float('inf')
    stop_counter = 0
    with tqdm(range(max_epochs), dynamic_ncols=True) as tqdmEpochs:
        for epoch in tqdmEpochs:
            train_loss = train_single(train_data, model, optimizer, device)
            val_loss = val_single(test_data, model, device)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy((model.state_dict()))
                torch.save(best_model_wts, model_filepath)
                stop_counter = 0
            else:
                stop_counter += 1

            tqdmEpochs.set_postfix(ordered_dict={
                "l": '%.2f' % train_loss + ',%.2f' % val_loss,
                'b': '%.2f' % best_loss,
                's': stop_counter
            })

            if stop_counter >= patience:
                break



def test_dl(test_data, time, data_name, pred_dict):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model_filepath = os.path.join('models', data_name, 'model_' + str(time))
    model.load_state_dict(torch.load(model_filepath))
    zero = torch.zeros((1, 5000)).to(device)
    one = torch.ones((1, 5000)).to(device)
    for inputs, labels, pids in test_data:
        inputs, labels = inputs.to(device), labels.to(device)
        out = model(inputs)
        pred = torch.where(out > 0.5, one, zero)
        for i in range(len(pids)):
            pid, lead = pids[i].split("_")
            if lead == 'i':
                pred_dict[pid] = {}
            pred_dict[pid][lead] = pred[i].cpu().detach().numpy()
    return pred_dict

