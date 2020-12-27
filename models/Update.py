#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb

class LocalUpdateMTL(object):
    def __init__(self, args, data_train, data_test):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.train_data = data_train
        self.test = data_test
        self.ldr_train = DataLoader(data_train, batch_size=self.args.batch_size, shuffle=True)
        #self.ldr_test = DataLoader(data_test, batch_size=len(data_test), shuffle=True)
        self.pretrain = False

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.5)

        epoch_loss = []
        if self.pretrain:
            local_eps = self.args.local_ep_pretrain
        else:
            local_eps = self.args.local_epochs

        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)

                W = W_glob.clone().to(self.args.device)

                W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                W_local = torch.cat(W_local)
                W[:, idx] = W_local.to(self.args.device)

                loss_regularizer = 0
                loss_regularizer += W.norm() ** 2

                #k = 1000
                k = W.shape[1]
                for i in range(k):
                    x = W[i * k:(i+1) * k, :]
                    loss_regularizer += x.mm(omega.to(self.args.device)).mm(x.T).trace()
                f = (int)(math.log10(W.shape[0])+1) + 1
                loss_regularizer *= 10 ** (-f)

                loss = loss + loss_regularizer
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
