#
# This code is adapted form paper
#
from comet_ml import Experiment
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
from utils.train_utils import get_model
from utils.options import args_parser
from models.Update import LocalUpdateMTL
from models.test import *
from utils.model_utils import *
import h5py
import pdb
from utils.plot_utils import *

if __name__ == '__main__':
    # parse args
    args = args_parser()
    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="multitask-learning",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : "MOCHA",
            "model":args.model,
            "batch_size":args.batch_size,
            "learning_rate":args.learning_rate,
            "beta" : args.beta, 
            "L_k" : args.L_k,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "optimizer": args.optimizer,
            "numusers": args.subusers,
            "K" : args.K,
            "personal_learning_rate" : args.personal_learning_rate,
            "times" : args.times,
            "gpu": args.gpu
        }
        experiment.log_parameters(hyper_params)
        experiment.set_name(args.dataset + "_" + "MOCHA" + "_" + args.model + "_" + str(args.batch_size) + "_" + str(args.learning_rate)+  "_" + str(args.num_global_iters) + "_"+ str(args.local_epochs) + "_"+ str(args.subusers))
    else:
        experiment = 0
    data = read_data(args.dataset)
    args.num_users = len(data[0])
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # build model
    for run_time in range(args.times):
        net_glob = get_model(args)
        net_glob.train()

        print(net_glob)
        net_glob.train()
        num_layers_keep  = 1
        total_num_layers = len(net_glob.weight_keys)
        w_glob_keys = net_glob.weight_keys[total_num_layers - num_layers_keep:]
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

        criterion = nn.NLLLoss()

        # training
        #results_save_path = os.path.join(base_save_dir, 'results.csv')

        loss_train = []
        net_best = None
        best_acc = np.ones(args.num_users) * -1
        best_net_list = copy.deepcopy(net_local_list)

        lr = args.learning_rate
        results = []
        local_list_users = []
        m = max(int(args.subusers * args.num_users), 1)
        I = torch.ones((m, m))
        i = torch.ones((m, 1))
        omega = I - 1 / m * i.mm(i.T)
        omega = omega ** 2
        omega.to(args.device)
        
        W = [net_local_list[0].state_dict()[key].flatten() for key in w_glob_keys]
        W = torch.cat(W)
        d = len(W)
        del W
        
        sub_data = args.cutoff
        total_users = len(data[0])
        print(total_users)
        if(sub_data):
            partion = int(0.9* total_users)
            randomList = np.random.choice(range(0, total_users), int(0.9*total_users), replace =False)
        
        #print(total_users,randomList)

        for idx in range(len(net_local_list)):
            id, train , test = read_user_data(idx, data, args.dataset)
            if(sub_data):
                if(idx in randomList):
                    train = train[int(0.95*len(train)):]
                    test   = test[int(0.8*len(test)):]
            local = LocalUpdateMTL(args=args, data_train = train, data_test = test)
            local_list_users.append(local)

        glob_acc = []
        train_acc = []
        train_loss = []
        avg_acc =[]

        for iter in range(args.num_global_iters):
            if(experiment):
                experiment.set_epoch(iter + 1)
            w_glob = {}
            loss_locals = []
            m = max(int(args.subusers * args.num_users), 1)
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
            acc_test_local_train, loss_test_local_train, acc_test_local_train_mean = test_img_local_all_train(net_local_list, args, local_list_users)
            acc_test_local_test, loss_test_local_test, acc_test_local_test_mean = test_img_local_all_test(net_local_list, args, local_list_users)
            glob_acc.append(acc_test_local_test)
            avg_acc.append(acc_test_local_test_mean)
            train_acc.append(acc_test_local_train)
            train_loss.append(loss_test_local_train)
            if(experiment):
                experiment.log_metric("glob_acc",acc_test_local_test)
                experiment.log_metric("avg_acc",acc_test_local_test_mean)
                experiment.log_metric("train_acc",acc_test_local_train)
                experiment.log_metric("train_loss",loss_test_local_train)
                
            print('Round {:4d}, Training Loss (local): {:.4f}, Training Acc (local): {:.4f} '.format(iter, loss_test_local_train, acc_test_local_train))
            print('Round {:4d}, Testing Loss (local): {:.4f}, Testing Acc (local): {:.4f}, Testing Acc (mean): {:.4f}'.format(iter, loss_test_local_test, acc_test_local_test, acc_test_local_test_mean))
        
        dir_path = "./results"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        alg = args.dataset + "_" + args.algorithm
        alg = alg + "_" + str(args.learning_rate) + "_" + str(args.beta) + "_" + str(args.L_k) + "_" + str(args.subusers) + "u" + "_" + str(args.batch_size) + "b" +  "_" + str(args.local_epochs)
        
        if(args.cutoff):
            alg = alg + "_"+ "subdata"
        alg = alg + "_" + str(run_time)
        
        if (len(glob_acc) != 0 &  len(train_acc) & len(train_loss)) :
            with h5py.File("./results/"+'{}.h5'.format(alg, args.local_epochs), 'w') as hf:
                hf.create_dataset('rs_avg_acc', data=avg_acc)
                hf.create_dataset('rs_glob_acc', data=glob_acc)
                hf.create_dataset('rs_train_acc', data=train_acc)
                hf.create_dataset('rs_train_loss', data=train_loss)
                hf.close()
                
    average_data(num_users=args.subusers, loc_ep1=args.local_epochs, Numb_Glob_Iters=args.num_global_iters, lamb=args.L_k,learning_rate=args.learning_rate, beta = args.beta, algorithms=args.algorithm, batch_size=args.batch_size, dataset=args.dataset, k = args.K, personal_learning_rate = args.personal_learning_rate,times = args.times,cutoff = args.cutoff)
