import torch
import os

from FLAlgorithms.users.userlocal import UserLocal
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np

class FedLocal(Server):
    def __init__(self,experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times, cutoff):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,
                         local_epochs, optimizer, num_users, times)

        # Initialize data for all  users
        self.sub_data = cutoff
        total_users = len(dataset[0][0])
        #np.random.seed(0)
        if(self.sub_data):
            partion = int(0.9* total_users)
            randomList = np.random.choice(range(0, total_users), int(0.9*total_users), replace =False)
            
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            if(self.sub_data):
                if(i in randomList):
                    train_ = train[int(0.95*len(train)):]
                    test_ = test[int(0.8*len(test)):]
                    user = UserLocal(device, id,train_ , test_, model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
            else:
                user = UserLocal(device, id, train, test, model, batch_size, learning_rate,beta,L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Number of users / total users:",num_users, " / " ,total_users)
        print("Finished creating Local model.")

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
                user.train()
            self.evaluate()
            #self.meta_evaluate()
        self.save_results()
        self.save_model()