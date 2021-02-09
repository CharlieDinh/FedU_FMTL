import torch
import os

from FLAlgorithms.users.userFedU import UserFedU
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class FedU(Server):
    def __init__(self,experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, K, times, cutoff):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        #subset data
        self.sub_data = cutoff
        self.data_set_name = dataset[1]
        self.K = K
        total_users = len(dataset[0][0])
        N = total_users
        b = np.random.uniform(0,1,size=(N,N))
        b_symm = (b + b.T)/2
        b_symm[b_symm < 0.25] = 0
        self.alk_connection = b_symm

        #np.random.seed(0)
        if(self.sub_data):
            randomList = self.get_partion(total_users)  
            
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)

            user = UserFedU(device, id, train, test, model, batch_size, learning_rate, beta, L_k, K,  local_epochs, optimizer)

            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:", num_users, " / " ,total_users)
        print("Finished creating SSGD server.")

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
        loss = []
        # only board cast one time
        self.send_parameters()
        #self.meta_split_users()
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            self.selected_users = self.select_users(glob_iter, self.num_users)
            # local update at each users
            for user in self.selected_users:
                user.train(self.local_epochs)
                
            # Agegrate parameter at each user
            if(self.L_k != 0): # if L_K = 0 it is local model 
                for user in self.selected_users:
                    user.aggregate_parameters(self.selected_users, glob_iter, len(self.users), self.data_set_name , self.alk_connection)
            self.evaluate()
            #self.meta_evaluate()
        self.save_results()
        self.save_model()