import torch
import torch.nn as nn
import torch.nn.functional as F
from FLAlgorithms.users.userbase import User
import numpy as np
# Implementation for AFL clients

FULL_BATCH = False

class UserAFL(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, L_k,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, L_k,
                         local_epochs)
        step_size = 60
        gamma = 1 # No learning rate decay, choose gamma in [0;1) for learning rate decay
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.schedule_optimizer = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=gamma)
        # print(f"step_size: {step_size}, gamma: {gamma}, full batch: {FULL_BATCH}")
    def train(self, epochs):
        # print("Training in userAFL!")
        LOSS = 0
        iter_num = 0
        self.model.train()
        if FULL_BATCH == True:
            for epoch in range(1, self.local_epochs + 1):
                for X,y in self.trainloaderfull:
                    X, y = X.to(self.device), y.long().to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(X), y)
                    loss.backward()
                    self.optimizer.step()
                    self.schedule_optimizer.step()
                    LOSS += loss
                    iter_num += 1
            # return LOSS, loss # return last loss value
            return LOSS, LOSS * 1.0 / iter_num
        else: # Running model with minibatch
            for epoch in range(1, self.local_epochs + 1):
                for X,y in self.trainloader:
                    X, y = X.to(self.device), y.long().to(self.device)
                    self.optimizer.zero_grad()
                    loss = self.loss(self.model(X), y)
                    loss.backward()
                    self.optimizer.step()
                    self.schedule_optimizer.step()
                    LOSS += loss
            return LOSS, loss # return last loss value

