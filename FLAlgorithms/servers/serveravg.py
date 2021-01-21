import torch
import os
import torch.multiprocessing as mp

from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

# Implementation for FedAvg Server

class FedAvg(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times , cutoff):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        total_users = len(dataset[0][0])
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            user = UserAVG(device, id, train, test, model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
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
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            #loss_ = 0
            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            self.selected_users = self.select_users(glob_iter,self.num_users)
            
            #NOTE: this is required for the ``fork`` method to work
            for user in self.selected_users:
                user.train(self.local_epochs)
            processes = []
            for user in self.selected_users:
                p = mp.Process(target=user.train(self.local_epochs))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            #for user in self.selected_users:
            #    user.train(self.local_epochs) #* user.train_samples


            self.aggregate_parameters()
            #loss_ /= self.total_train_samples
            #loss.append(loss_)
            #print(loss_)
        #print(loss)
        self.save_results()
        self.save_model()