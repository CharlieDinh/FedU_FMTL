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

class UserFedU(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k, K, local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, L_k,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()
        self.K = K
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay = 0)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs):
        LOSS = 0
        self.model.train()
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

    def aggregate_parameters(self, user_list, global_in, num_clients, dataset, alk_connection):
        avg_weight_different = copy.deepcopy(list(self.model.parameters()))
        akl = alk_connection
        for param in avg_weight_different:
            param.data = torch.zeros_like(param.data)
        
        # Calculate the diffence of model between all users or tasks
        #for i in range(len(user_list)):
        #    if(self.id != user_list[i].id):
        #        if(self.K > 0 and self.K <= 2):
        #            akl[int(self.id)][int(user_list[i].id)] = self.get_alk(user_list, dataset, i)
                # K == 3 : akl will be generate randomly for MNIST
        #        for avg, current_task, other_tasks in zip(avg_weight_different, self.model.parameters(),user_list[i].model.parameters()):
        #            avg.data += akl[int(self.id)][int(user_list[i].id)] * (current_task.data.clone() - other_tasks.data.clone())
        
        for avg, current_task in zip(avg_weight_different, self.model.parameters()):
            current_task.data = current_task.data - 0.5 * self.learning_rate * self.L_k * self.beta * self.local_epochs * avg
