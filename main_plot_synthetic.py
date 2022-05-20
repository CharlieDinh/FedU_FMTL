#!/usr/bin/env python
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from utils.plot_utils import *
import torch
torch.manual_seed(0)

if(1): # plot for Mnist
    numusers = 0.1
    num_glob_iters = 200
    dataset = "Synthetic"
    local_ep = [1,1,1,1,1]
    L_k = [0.004,0.004,0.004,0.004]
    learning_rate = [0.003, 0.003, 0.003, 0.003, 0.003]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0, 0, 0, 0]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = ["FedU","pFedMe","FedAvg","SCAFFOLD","AFL"]
    plot_summary_synthetic(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
    learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
