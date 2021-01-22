#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="human_activity", choices=["EMNIST","human_activity", "gleam","vehicle_sensor","Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="mclr", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--L_k", type=float, default=1, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default = 5)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="SSGD",choices=["pFedMe", "pFedMe_p", "PerAvg", "FedAvg", "SSGD", "Mocha", "Local" , "Global"]) 
    parser.add_argument("--subusers", type = float, default = 1, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=0, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to commet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments")
    parser.add_argument("--cutoff", type=int, default=0, help="Cutoff data sample")
    args = parser.parse_args()

    args = parser.parse_args()
    return args
