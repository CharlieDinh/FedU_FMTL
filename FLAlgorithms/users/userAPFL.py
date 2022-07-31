import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
import numpy as np
import copy
# Implementation for APFL clients


class UserAPFL(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, L_k,
                         local_epochs)
        
        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.alpha = beta # using beta as alpha parameter
        self.apfl_v = copy.deepcopy(list(self.model.parameters()))

    def train(self, epochs):
        # print("Training in userAFL!")
        LOSS = 0
        iter_num = 0
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            for X1,y1 in self.trainloader:
                X, y = X1.to(self.device), y1.long().to(self.device)

                # update local model w^(t)?
                self.set_local_parameters(self.local_model)
                self.optimizer.zero_grad()
                loss_w = self.loss(self.model(X), y)
                loss_w.backward()
                self.optimizer.step()

                # update persionalized model
                self.clone_model_paramenter(self.model.parameters(), self.local_model)
                self.set_local_parameters(self.apfl_v)
                self.optimizer.zero_grad()
                loss_v = self.loss(self.model(X), y)
                loss_v.backward()
                self.optimizer.step()

                for v_weight, local_weight in zip(self.apfl_v, self.local_model):
                    v_weight.data = self.alpha * v_weight.data + (1-self.alpha)*local_weight.data
                    
                LOSS += loss_w
        # update persionalized model
        self.clone_model_paramenter(self.apfl_v,self.persionalized_model_bar)

        return LOSS, loss_w # return last loss value






