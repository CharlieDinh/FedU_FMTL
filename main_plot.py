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

if(): # plot for Human Activity
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "human_activity"
    local_ep = [5,5,5,5,5]
    L_k = [1.0,0.1,0.01,0.001,0.01]
    learning_rate = [0.03, 0.03, 0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20]
    K = [0,0,0,0,0,0]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local"]#,"PerAvg_p","FedAvg"]
    plot_summary_human_activity_eta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(1): # plot for Human Activity
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "vehicle_sensor"
    local_ep = [5,5,5,5,5]
    L_k = [1.0,0.1,0.01,0.001,0.01]
    learning_rate = [0.05, 0.05, 0.05, 0.05, 0.05]
    beta =  [1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20]
    K = [0,0,0,0,0,0]
    personal_learning_rate = [0.09,0.09,0.09,0.09,0.09]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local"]#,"PerAvg_p","FedAvg"]
    plot_summary_vehicle_eta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)