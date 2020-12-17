#!/usr/bin/env python
from comet_ml import Experiment
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
import importlib
import random
import os
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.servermavg import mFedAvg
from FLAlgorithms.servers.serverpFedMe import pFedMe
from FLAlgorithms.servers.servermpFedMe import mpFedMe
from FLAlgorithms.servers.serverperavg import PerAvg
from FLAlgorithms.servers.serverSSGD import FedSSGD
from FLAlgorithms.servers.servermSSGD import mFedSSGD
from FLAlgorithms.servers.servermSSGD3 import mFedSSGD3
from utils.model_utils import read_data
from FLAlgorithms.trainmodel.models import *
from utils.plot_utils import *
import torch
torch.manual_seed(0)

# import comet_ml at the top of your file


# Create an experiment with your api key:
def main(experiment, dataset, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters,
         local_epochs, optimizer, numusers, K, personal_learning_rate, times, commet, gpu):
    
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    data = read_data(dataset) , dataset

    for i in range(times):
        print("---------------Running time:------------",i)
        # Generate model
        if(model == "mclr"):
            if(dataset == "human_activity"):
                model = Mclr_Logistic(561,6).to(device), model
            elif(dataset == "gleam"):
                model = Mclr_Logistic(561,6).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = Mclr_Logistic(100,2).to(device), model
            elif(dataset == "Synthetic"):
                model = Mclr_Logistic(60,10).to(device), model
            else:#(dataset == "Mnist"):
                model = Mclr_Logistic().to(device), model

        if(model == "dnn"):
            if(dataset == "human_activity"):
                model = DNN2(561,100,100,12).to(device), model
            elif(dataset == "gleam"):
                model = DNN(561,20,6).to(device), model
            elif(dataset == "vehicle_sensor"):
                model = DNN(100,20,2).to(device), model
            elif(dataset == "Synthetic"):
                model = DNN(60,20,10).to(device), model
            else:#(dataset == "Mnist"):
                model = DNN2().to(device), model
        
        if(model == "cnn"):
            if(dataset == "Cifar10"):
                model = CifarNet().to(device), model

        # select algorithm

        if(algorithm == "FedAvg"):
            if(commet):
                experiment.set_name("cloud_nongpu"+dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedAvg(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "PerAvg"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(personal_learning_rate) + "_" + str(learning_rate)+  "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = PerAvg(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)

        if(algorithm == "SSGD"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = FedSSGD(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "mFedAvg"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = mFedAvg(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "pFedMe"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(personal_learning_rate) + "_" + str(learning_rate)+  "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = pFedMe(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i)
        
        if(algorithm == "mpFedMe"):
            if(commet):
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(personal_learning_rate) + "_" + str(learning_rate)+  "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = mpFedMe(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, K, personal_learning_rate, i)

        if(algorithm == "mSSGD"):
            if(commet): 
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = mFedSSGD(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)
        
        if(algorithm == "mSSGD3"):
            if(commet): 
                experiment.set_name(dataset + "_" + algorithm + "_" + model[1] + "_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(L_k) + "_" + str(num_glob_iters) + "_"+ str(local_epochs) + "_"+ str(numusers))
            server = mFedSSGD3(experiment, device, data, algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, numusers, i)

        server.train()
        server.test()

    average_data(num_users=numusers, loc_ep1=local_epochs, Numb_Glob_Iters=num_glob_iters, lamb=L_k,learning_rate=learning_rate, beta = beta, algorithms=algorithm, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate,times = times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cifar10", choices=["human_activity", "gleam","vehicle_sensor","Mnist", "Synthetic", "Cifar10"])
    parser.add_argument("--model", type=str, default="cnn", choices=["dnn", "mclr", "cnn"])
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--beta", type=float, default=1, help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--L_k", type=int, default=15, help="Regularization term")
    parser.add_argument("--num_global_iters", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--algorithm", type=str, default="PerAvg",choices=["pFedMe", "PerAvg", "FedAvg", "SSGD", "mSSGD","mFedAvg"]) 
    parser.add_argument("--numusers", type=int, default=20, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=5, help="Computation steps")
    parser.add_argument("--personal_learning_rate", type=float, default=0.01, help="Persionalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--commet", type=int, default=0, help="log data to commet")
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to run the experiments")
    args = parser.parse_args()

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.numusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    print("Local Model       : {}".format(args.model))
    print("=" * 80)

    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="multitask-learning",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "model":args.model,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "beta" : args.beta, 
            "L_k" : args.L_k,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "optimizer": args.optimizer,
            "numusers": args.numusers,
            "K" : args.K,
            "personal_learning_rate" : args.personal_learning_rate,
            "times" : args.times,
            "gpu": args.gpu
        }
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        model=args.model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta = args.beta, 
        L_k = args.L_k,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        optimizer= args.optimizer,
        numusers = args.numusers,
        K=args.K,
        personal_learning_rate=args.personal_learning_rate,
        times = args.times,
        commet = args.commet,
        gpu=args.gpu
        )


