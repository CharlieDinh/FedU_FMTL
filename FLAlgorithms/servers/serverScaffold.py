import torch
import os

import h5py
from FLAlgorithms.users.userScaffold import UserSCAFFOLD
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
from scipy.stats import rayleigh


# Implementation for SCAFFOLD Server
class SCAFFOLD(Server):
    def __init__(self, experiment, device, dataset,algorithm, model, batch_size, learning_rate, beta, L_k, num_glob_iters, local_epochs, optimizer, num_users, times , cutoff):
        super().__init__(experiment, device, dataset,algorithm, model[0], batch_size, learning_rate, beta, L_k, num_glob_iters,local_epochs, optimizer, num_users, times)
        
        self.control_norms = []
        self.noise = 0
        self.communication_thresh = None
        self.param_norms = []
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
            user = UserSCAFFOLD(device, id, train, test, model, batch_size, learning_rate, beta, L_k, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        if self.noise:
            self.communication_thresh = rayleigh.ppf(1 - num_users)  # h_min

        print("Number of users / total users:", num_users*total_users, " / ", total_users)

        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]

        print("Finished creating SCAFFOLD server.")

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0

            self.send_parameters()

            # Evaluate model at each iteration
            self.evaluate()

            if self.noise:
                self.selected_users = self.select_transmitting_users()
                print(f"Transmitting {len(self.selected_users)} users")
            else:
                self.selected_users = self.select_users(glob_iter, self.num_users)

            for user in self.selected_users:
                user.train()
                user.drop_lr()

            self.aggregate_parameters()
            self.get_max_norm()

            if self.noise:
                self.apply_channel_effect()

        self.save_results()
        #self.save_norms()
        self.save_model()

    def send_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)
            for control, new_control in zip(user.server_controls, self.server_controls):
                control.data = new_control.data

    def aggregate_parameters(self):
        assert (self.users is not None and len(self.users) > 0)
        total_samples = 0
        for user in self.selected_users:
            total_samples += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, total_samples)

    def add_parameters(self, user, total_samples):
        num_of_selected_users = len(self.selected_users)
        num_of_users = len(self.users)
        num_of_samples = user.train_samples
        for param, control, del_control, del_model in zip(self.model.parameters(), self.server_controls,
                                                          user.delta_controls, user.delta_model):
            # param.data = param.data + del_model.data * num_of_samples / total_samples / num_of_selected_users
            param.data = param.data + del_model.data / num_of_selected_users
            control.data = control.data + del_control.data / num_of_users

    def get_max_norm(self):
        param_norms = []
        control_norms = []
        for user in self.selected_users:
            param_norm, control_norm = user.get_params_norm()
            param_norms.append(param_norm)
            control_norms.append(control_norm)
        self.param_norms.append(max(param_norms))
        self.control_norms.append((max(control_norms)))

    def apply_channel_effect(self, sigma=1, power_control=2500):
        num_of_selected_users = len(self.selected_users)
        alpha_t_params = power_control / self.param_norms[-1] ** 2
        alpha_t_controls = 4e4 * power_control / self.control_norms[-1] ** 2
        for param, control in zip(self.model.parameters(), self.server_controls):
            param.data = param.data + sigma / (
                        alpha_t_params ** 0.5 * num_of_selected_users * self.communication_thresh) * torch.randn(
                param.data.size())
            control.data = control.data + sigma / (
                        alpha_t_controls ** 0.5 * num_of_selected_users * self.communication_thresh) * torch.randn(
                control.data.size())
