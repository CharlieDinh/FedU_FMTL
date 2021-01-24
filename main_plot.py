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


#------------------------------------------------------Human Activity------------------------------------------------------
if(0): # plot for Human Activity
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "human_activity"
    local_ep = [5,5,5,5,5,5,5]
    L_k = [0.001,0.01,0.1,1.0,0.01,0.01,100.0]
    learning_rate = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,0,0,0,0,0,0,0]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local", "Global", "Mocha"]#,"PerAvg_p","FedAvg"]
    plot_summary_human_activity_eta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)


if(0): # plot for Human Activity
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "human_activity"
    local_ep = [5,5,5]
    L_k = [0.1,0.1,0.1]
    learning_rate = [0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,1,2]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD"]#,"PerAvg_p","FedAvg"]
    plot_summary_human_activity_akl(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # plot for Human Activity
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "human_activity"
    local_ep = [5,5,5,5,5,5,5]
    L_k = [0.001,0.01,0.1,1.0,0.01,0.01,100.0]
    learning_rate = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,0,0,0,0,0,0,0]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local", "Global", "Mocha"]#,"PerAvg_p","FedAvg"]
    plot_summary_human_activity_eta_non(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)


if(0): # plot for Human Activity
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "human_activity"
    local_ep = [5,5,5]
    L_k = [0.1,0.1,0.1]
    learning_rate = [0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,1,2]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD"]#,"PerAvg_p","FedAvg"]
    plot_summary_human_activity_akl_non(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

#------------------------------------------------------vehicle_sensor------------------------------------------------------
if(0): # plot for vehicle_sensor
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "vehicle_sensor"
    local_ep = [5,5,5,5,5,5,5]
    L_k = [0.001,0.01,0.1,1.0,0.01,0.01,100.0]
    learning_rate = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,0,0,0,0,0,0]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local", "Global", "Mocha"]#,"PerAvg_p","FedAvg"]
    plot_summary_vehicle_eta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(1): # plot for vehicle_sensor
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "vehicle_sensor"
    local_ep = [5,5,5,5,5,5,5]
    L_k = [0.01,0.01,0.01]
    learning_rate = [0.05, 0.05, 0.05, 0.05, 0.05]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,1,2]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD"]#,"PerAvg_p","FedAvg"]
    plot_summary_vehicle_akl(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # plot for vehicle_sensor
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "vehicle_sensor"
    local_ep = [5,5,5,5,5,5,5]
    L_k = [0.001,0.01,0.1,1.0,0.01,0.01,100.0]
    learning_rate = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,0,0,0,0,0,0]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local", "Global", "Mocha"]#,"PerAvg_p","FedAvg"]
    plot_summary_vehicle_eta_non(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # plot for vehicle_sensor
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "vehicle_sensor"
    local_ep = [5,5,5,5,5]
    L_k = [0.01,0.01,0.01]
    learning_rate = [0.05, 0.05, 0.05, 0.05, 0.05]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,1,2]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD"]#,"PerAvg_p","FedAvg"]
    plot_summary_vehicle_akl_non(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

#------------------------------------------------------Mnist------------------------------------------------------

if(0):
    numusers = 1.0 # Mnist
    num_glob_iters = 200
    dataset = "Mnist"
    local_ep = [5,5,5,5,5,5,5]
    L_k = [0.001,0.005,0.01,0.05,0.01,0.01,100.0]
    learning_rate = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,0,0,0,0,0,0]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local", "Global", "Mocha"]#,"PerAvg_p","FedAvg"]
    plot_summary_mnist_eta(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # plot for Mnist
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "Mnist"
    local_ep = [5,5,5,5,5]
    L_k = [0.005,0.005,0.005,0.005]
    learning_rate = [0.03, 0.03, 0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,1,2,3]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD"]#,"PerAvg_p","FedAvg"]
    plot_summary_mnist_akl(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
if(0):
    numusers = 1.0 # Mnist
    num_glob_iters = 200
    dataset = "Mnist"
    local_ep = [5,5,5,5,5,5,5]
    L_k = [0.001,0.005,0.01,0.05,0.01,0.01,100.0]
    learning_rate = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0,0,0,0,0,0,0]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD", "Local", "Global", "Mocha"]#,"PerAvg_p","FedAvg"]
    plot_summary_mnist_eta_non(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)

if(0): # plot for Mnist
    numusers = 1.0 # full user
    num_glob_iters = 200
    dataset = "Mnist"
    local_ep = [5,5,5,5,5]
    L_k = [0.005,0.005,0.005,0.005]
    learning_rate = [0.03, 0.03, 0.03, 0.03, 0.03]
    beta =  [1, 1, 1, 1, 1, 1, 1]
    batch_size = [20,20,20,20,20,20,20]
    K = [0, 1, 2, 3]
    personal_learning_rate = [1,1,1,1,1,1,1,1]
    algorithms = [ "SSGD","SSGD","SSGD","SSGD"]#,"PerAvg_p","FedAvg"]
    plot_summary_mnist_akl_non(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=L_k,
                               learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
