#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import copy
import numpy as np
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pdb

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def test_img(net_g, datatest, args, return_probs=False, user_idx=-1):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    probs = []

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        probs.append(log_probs)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    # if args.verbose:
    #     if user_idx < 0:
    #         print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #             test_loss, correct, len(data_loader.dataset), accuracy))
    #     else:
    #         print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #             user_idx, test_loss, correct, len(data_loader.dataset), accuracy))

    if return_probs:
        return accuracy, test_loss, torch.cat(probs)
    return accuracy, test_loss


def test_img_local_test(net_g, args, user):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(user.test, batch_size=len(user.test), shuffle=False)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    #accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    # if args.verbose:
    #     print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(data_loader.dataset), accuracy))

    return accuracy, test_loss, correct


def test_img_local_train(net_g, args, user):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(dataset, batch_size=args.bs)
    data_loader = DataLoader(user.train_data, batch_size=len(user.train_data), shuffle=False)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)

        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    #accuracy = 100.00 * float(correct) / len(data_loader.dataset)
    # if args.verbose:
    #     print('Local model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, correct


def test_img_local_all_test(net_local_list, args, local_list_users, return_all= False):
    correct_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    acc_test_local = np.zeros(args.num_users)
    total_test_sample = 0
    total_correct = 0
    for idx in range(args.num_users):
        total_test_sample += len(local_list_users[idx].test)
        net_local = net_local_list[idx]
        net_local.eval()
        a, b, c = test_img_local_test(net_local, args, local_list_users[idx])
        correct_test_local[idx] = c
        acc_test_local[idx] = a
        loss_test_local[idx] = b
        total_correct += c
    if return_all:
        return acc_test_local, loss_test_local
    else: 
        return float(total_correct)/total_test_sample, loss_test_local.mean() , acc_test_local.mean()
        #return acc_test_local.mean(), loss_test_local.mean()

def test_img_local_all_train(net_local_list, args, local_list_users, return_all= False):
    correct_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)
    acc_test_local = np.zeros(args.num_users)
    total_test_sample = 0
    total_correct = 0
    for idx in range(args.num_users):
        total_test_sample += len(local_list_users[idx].train_data)
        net_local = net_local_list[idx]
        net_local.eval()
        a, b, c = test_img_local_train(net_local, args, local_list_users[idx])
        correct_test_local[idx] = c
        acc_test_local[idx] = a
        loss_test_local[idx] = b
        total_correct += c
    if return_all:
        return acc_test_local, loss_test_local
    else: 
        return float(total_correct)/total_test_sample, loss_test_local.mean(), acc_test_local.mean()
        #return acc_test_local.mean(), loss_test_local.mean()

def test_img_avg_all(net_glob, net_local_list, args, local_list_users, return_net=False):

    net_glob_temp = copy.deepcopy(net_glob)
    w_keys_epoch = net_glob.state_dict().keys()
    w_glob_temp = {}

    for idx in range(args.num_users):
        net_local = net_local_list[idx]
        w_local = net_local.state_dict()

        if len(w_glob_temp) == 0:
            w_glob_temp = copy.deepcopy(w_local)
        else:
            for k in w_keys_epoch:
                w_glob_temp[k] += w_local[k]

    for k in w_keys_epoch:
        w_glob_temp[k] = torch.div(w_glob_temp[k], args.num_users)

    net_glob_temp.load_state_dict(w_glob_temp)
    acc_test_avg, loss_test_avg = test_img_global(net_glob_temp, args,  local_list_users)
    return acc_test_avg, loss_test_avg
