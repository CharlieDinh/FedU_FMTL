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

if(1): # plot for MNIST convex 
    numusers = 30
    num_glob_iters = 200
    dataset = "vehicle_sensor"
    local_ep = [20,20,20,20]
    L_k = [0,0,0,0]
    learning_rate = [0.001, 0.001, 0.001, 0.001]
    beta =  [1.0, 1.0, 0.001, 1.0]
    batch_size = [20,20,20,20]
    K = [5,5,5,5,5,5]
    personal_learning_rate = [0.09,0.09,0.09,0.09]
    algorithms = [ "SSGD","FedAvg"]#,"PerAvg_p","FedAvg"]
    plot_summary_one_figure(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)