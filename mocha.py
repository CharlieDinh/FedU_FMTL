#
# This code is adapted form paper
#
import copy
import os
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import mode
from torchvision import datasets, transforms, models
import torch
from torch import nn

from utils.train_utils import get_model, get_data
from utils.options import args_parser
from models.Update import LocalUpdateMTL
from models.test import test_img, test_img_local, test_img_local_all, test_img_avg_all

from utils.model_utils import read_data, read_user_data

import pdb

if __name__ == '__main__':
    # parse args
    args = args_parser()
    
    data = read_data(args.dataset)
    args.num_users = len(data[0])
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # build model
    net_glob = get_model(args)
    net_glob.train()

    print(net_glob)
    net_glob.train()

    total_num_layers = len(net_glob.weight_keys)
    w_glob_keys = net_glob.weight_keys[total_num_layers - args.num_layers_keep:]
    w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

    num_param_glob = 0
    num_param_local = 0
    for key in net_glob.state_dict().keys():
        num_param_local += net_glob.state_dict()[key].numel()
        if key in w_glob_keys:
            num_param_glob += net_glob.state_dict()[key].numel()
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))

    # generate list of local models for each user
    net_local_list = []
    for user_ix in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))

    criterion = nn.CrossEntropyLoss()

    # training
    #results_save_path = os.path.join(base_save_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_acc = np.ones(args.num_users) * -1
    best_net_list = copy.deepcopy(net_local_list)

    lr = args.lr
    results = []
    local_list_users = []
    m = max(int(args.frac * args.num_users), 1)
    I = torch.ones((m, m))
    i = torch.ones((m, 1))
    omega = I - 1 / m * i.mm(i.T)
    omega = omega ** 2
    #omega = omega.cuda()
    omega.to(args.device)
    
    W = [net_local_list[0].state_dict()[key].flatten() for key in w_glob_keys]
    W = torch.cat(W)
    d = len(W)
    del W

    for idx in range(len(net_local_list)):
            id, train , test = read_user_data(idx, data, args.dataset)
            local = LocalUpdateMTL(args=args, data_train = train, data_test = test)
            local_list_users.append(local)

    for iter in range(args.epochs):
        w_glob = {}
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        W = torch.zeros((d, m)).to(args.device)#.cuda()

        # update W
        for idx, user in enumerate(idxs_users):
            W_local = [net_local_list[user].state_dict()[key].flatten() for key in w_glob_keys]
            W_local = torch.cat(W_local)
            W[:, idx] = W_local

        # update local model
        for idx, user in enumerate(idxs_users):
            w_local, loss = local_list_users[user].train(net=net_local_list[user].to(args.device), lr=lr, omega=omega, W_glob=W.clone(), idx=idx, w_glob_keys=w_glob_keys)

        # evaluate local model
        acc_test_local, loss_test_local = test_img_local_all(net_local_list, args, local_list_users)
        
        print('Round {:3d}, Loss (local): {:.3f}, Acc (local): {:.2f}'.format(iter, loss_test_local, acc_test_local))