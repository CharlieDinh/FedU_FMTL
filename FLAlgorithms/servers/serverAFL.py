import torch
from FLAlgorithms.users.userAFL import UserAFL
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import copy
# Implementation for FedAvg Server

AFL = True 
AFL_GRAD = False
CHECK_AVG_PARAM = False

class FedAFL(Server):
    def __init__(self,experiment, device, dataset,algorithm, model, batch_size, learning_rate, gamma, L_k, num_glob_iters, local_epochs, optimizer, num_users, times, cutoff):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, gamma, L_k, num_glob_iters, local_epochs, optimizer, num_users, times)

        # Grads
        self.grad_learning_rate = learning_rate
        # Initialize lambdas

       

        total_users = len(dataset[0][0])
        lamdas_length = int(num_users * total_users) # all clients will be involved in selection for training

        self.lambdas = np.ones(lamdas_length) * 1.0 /(lamdas_length)
        self.learning_rate_lambda = 0.001
        print(f"lambdas learning rate: {self.learning_rate_lambda}")
        self.K = 0

        self.sub_data = cutoff
        if(self.sub_data):
            randomList = self.get_partion(total_users)     
        for i in range(total_users):
            id, train , test = read_user_data(i, dataset[0], dataset[1])
            if(self.sub_data):
                if(i in randomList):
                    train, test = self.get_data(train, test)
            user = UserAFL(device, id, train, test, model, batch_size, learning_rate, gamma, L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

            
        print("Number of users / total users:", num_users, " / " ,total_users)
        print("Finished creating AFL server.")

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

    def project(self, y):
        ''' algorithm comes from:
        https://arxiv.org/pdf/1309.1541.pdf
        '''
        if np.sum(y) <=1 and np.alltrue(y >= 0):
            return y
        u = sorted(y, reverse=True)
        x = []
        rho = 0
        for i in range(len(y)):
            if (u[i] + (1.0/(i+1)) * (1-np.sum(np.asarray(u)[:i]))) > 0:
                rho = i + 1
        lambda_ = (1.0/rho) * (1-np.sum(np.asarray(u)[:rho]))
        for i in range(len(y)):
            x.append(max(y[i]+lambda_, 0))
        return x  

    def train(self):
        # Assign all values of resulting model = 0
        for resulting_model_param in self.resulting_model.parameters():
            resulting_model_param.data = torch.zeros_like(resulting_model_param.data)
        # Training for clients
        for glob_iter in range(self.num_glob_iters):
            losses = []
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: AFL", glob_iter, " -------------")

            self.send_parameters()

            # Evaluate model each interation
            self.evaluate()

            # Select subset of user for training
            self.selected_users = self.select_users(glob_iter,self.num_users)

            for user in self.selected_users:
                _, loss = user.train(self.local_epochs)
                if AFL==True:
                    losses.append(loss.data.item()) # Collect loss from users
                     # print(f"losses: {losses}")

            if AFL == True: # Select AFL algorithms
                # Aggregate training result from users
                if AFL_GRAD == True: # Aggregate based on gradients
                    print("AFL Grads!!!")
                    self.AFL_aggregate_grads(self.selected_users, self.lambdas)
                    # Projection weights
                    for server_param in self.model.parameters():
                        server_param.data -= server_param.grad * self.grad_learning_rate
                else: # Aggregate based on weights
                    print("AFL Weights!!!")
                    self.AFL_aggregate_parameters(self.selected_users, self.lambdas)

                # Update lambdas
                for idx in range(len(self.lambdas)):
                    self.lambdas[idx] += self.learning_rate_lambda * losses[idx]
                # Project lambdas
                self.lambdas = self.project(self.lambdas)
                # Avoid probability 0
                self.lambdas = np.asarray(self.lambdas)
            else:
                # FedAvg
                self.aggregate_parameters(self.selected_users)

            # Check averaged weights
            if CHECK_AVG_PARAM == True:
                for server_param in self.model.parameters():
                    print(server_param.data)

            # Averaging model
            for server_param, resulting_model_param in zip(self.model.parameters(), self.resulting_model.parameters()):
                resulting_model_param.data = (resulting_model_param.data*glob_iter + server_param.data) * 1.0 / (glob_iter + 1)

        # Distribute the final model to all users        
        print(f"-----Testing on final model-----")
        for server_param, resulting_model_param in zip(self.model.parameters(), self.resulting_model.parameters()):
            server_param.data = resulting_model_param.data
            if CHECK_AVG_PARAM == True:
                print(server_param.data)
        self.send_parameters()
        self.evaluate()

        self.save_results()
        self.save_model()