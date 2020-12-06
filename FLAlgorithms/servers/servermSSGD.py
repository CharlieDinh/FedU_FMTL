import torch
import os

from FLAlgorithms.users.usermSSGD import UsermSSGD
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class mFedSSGD(Server):
    def __init__(self,experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        total_users = len(dataset[0][0])
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            user = UsermSSGD(device, id, train, test, model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating FedAvg server.")

    def send_grads(self):
        assert (self.users is not None and len(self.users) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.users:
            user.set_grads(grads)

    def train(self):
        self.send_parameters()
        self.meta_split_users()

        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")

            # For training process
            # local update at each users
            
            for user in self.selected_train:
                user.train(self.local_epochs)
            # Agegrate parameter at each user 
            selected_train = self.select_sub_train_users(self.num_users)
            for user in self.selected_train:
                #user.aggregate_parameters(self.train_users)
                user.aggregate_parameters2(self.train_users,self.model.parameters())
            
            self.aggregate_meta_parameters()
            # send meta model to all 
            self.send_meta_parameters_totest()

            # For testing meta model 
            # For testing
            for user in self.test_users:
                # reset to initial point
                #for param in user.model.parameters():
                    #param.data = torch.zeros_like(param.data)
                user.train(5)
            # Agegrate parameter at each user 
            for user in self.test_users:
                user.aggregate_parameters(self.train_users)
            
            self.meta_evaluate()
            
            
        self.save_results()
        self.save_model()