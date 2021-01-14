import torch
import os

from FLAlgorithms.users.userglobal import UserGlobal
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

class FedGlobal(Server):
    def __init__(self,experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times)


        self.sub_data = 0
        total_users = len(dataset[0][0])
        #np.random.seed(0)
        if(self.sub_data):
            partion = int(0.9* total_users)
            randomList = np.random.choice(range(0, total_users), int(0.9*total_users), replace =False)
            
        # Initialize data for all  users
        total_users = len(dataset[0][0])
        train_all = []
        test_all = []
        
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            if(self.sub_data):
                if(i in randomList):
                    train_ = train[int(0.95*len(train)):]
                    test_ = test[int(0.8*len(test)):]
                    train_all += train_
                    test_all += test_
            else:
                train_all += train
                test_all += test

        id = "0001"
        
        user = UserGlobal(device, id, train_all, test_all, model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
        self.users.append(user)
        self.total_train_samples = len(train_all)
            
        print("Finished creating Global model.")

    def train(self):
        loss = []
        # only board cast one time
        self.send_parameters()
        #self.meta_split_users()
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")
            self.users[0].train()
            self.evaluate()
            #self.meta_evaluate()
        self.save_results()
        self.save_model()