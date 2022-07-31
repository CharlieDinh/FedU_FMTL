import torch
from FLAlgorithms.users.userAPFL import UserAPFL
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
# Implementation for FedAvg Server

class FedAPFL(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times , cutoff):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,local_epochs, optimizer, num_users, times)
        # Initialize data for all  users
        total_users = len(dataset[0][0])
        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(total_users)  
            
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)
                    
            user = UserAPFL(device, id, train, test, model, batch_size, learning_rate, beta, L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating APFL server.")


    def train(self):
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            # send all parameter for users 
            self.send_parameters()

            # Evaluate gloal model on user for each interation
            print("Evaluate global model")
            self.evaluate()

            for user in self.users:
                user.train(self.local_epochs) 
            self.selected_users = self.select_users(glob_iter,self.num_users)

            # Evaluate gloal model on user for each interation
            print("Evaluate persionalized model")
            self.evaluate_personalized_model()
            self.aggregate_parameters()
            
        self.save_results()
        self.save_model()