import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User
import copy
import numpy as np
# Implementation for FedAvg clients

class UserSSGD(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k, local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, L_k,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay = 0)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    # def train(self, epochs):
    #     LOSS = 0
    #     self.model.train()
    #     for epoch in range(1, self.local_epochs + 1):
    #         self.model.train()
    #         X, y = self.get_next_train_batch()
    #         self.optimizer.zero_grad()
    #         output = self.model(X)
    #         loss = self.loss(output, y)
    #         loss.backward()
    #         self.optimizer.step()
    #         # local model is perci
    #         self.clone_model_paramenter(self.model.parameters(), self.local_model)
    #     return LOSS
    
    def train(self, epochs):
        LOSS = 0
        self.model.train()
        #self.clone_model_paramenter(self.model.parameters(), self.local_model)
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for X,y in self.trainloader:
                X, y = X.to(self.device), y.to(self.device)#self.get_next_train_batch()
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return LOSS

    def aggregate_parameters(self, user_list, global_in, num_clients):
        avg_weight_different = copy.deepcopy(list(self.model.parameters()))
        alpha = np.ones(len(user_list))
        for param in avg_weight_different:
            param.data = torch.zeros_like(param.data)
        
        # Calculate the diffence of model between all users or tasks
        for i in range(len(user_list)):
            if(self.id != user_list[i].id):
                for avg, current_task, other_tasks in zip(avg_weight_different, self.model.parameters(),user_list[i].model.parameters()):
                    avg.data += alpha[i] * (current_task.data.clone() - other_tasks.data.clone())
        
        for avg, current_task in zip(avg_weight_different, self.model.parameters()):
            current_task.data = current_task.data - self.learning_rate * self.L_k *avg
        
        #beta = (float)(2 * num_clients)/ (global_in + 2 * num_clients)

        # update current task follow 15 rulle
        #for local, current_task in zip(self.local_model, self.model.parameters()):
        #    current_task.data = (1 - beta)*local.data + beta * current_task.data
        #self.clone_model_paramenter(self.model.parameters(), self.local_model)
        



