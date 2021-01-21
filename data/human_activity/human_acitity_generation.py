import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import random
import json

random.seed(1)

NUM_USERS = 30 
NUM_LABELS = 6

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)
dir_path_train = os.path.join(dir_path, 'human_activity/Original_Data/train')
dir_path_test = os.path.join(dir_path, 'human_activity/Original_Data/test')

x_train = pd.read_csv(os.path.join(dir_path_train, 'X_train.txt'), delimiter='\n', header=None).values
y_train = pd.read_csv(os.path.join(dir_path_train, 'y_train.txt'), delimiter='\n', header=None).values
task_index_train = pd.read_csv(os.path.join(dir_path_train, 'subject_train.txt'), delimiter='\n', header=None).values
x_test = pd.read_csv(os.path.join(dir_path_test, 'X_test.txt'), delimiter='\n', header=None).values
y_test = pd.read_csv(os.path.join(dir_path_test, 'y_test.txt'), delimiter='\n', header=None).values
task_index_test = pd.read_csv(os.path.join(dir_path_test, 'subject_test.txt'), delimiter='\n', header=None).values

for i in range(len(x_train)):
    x_train[i][0] = [float(x) for x in x_train[i][0].split()]
for i in range(len(x_test)):
    x_test[i][0] = [float(x) for x in x_test[i][0].split()]

train_path = './data/train/human_train.json'
test_path = './data/test/human_test.json'

dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    
X_Con = np.concatenate((x_train, x_test))
y_Con = np.concatenate((y_train, y_test)).squeeze()
y_Con = y_Con - 1 
task_index = np.concatenate((task_index_train, task_index_test)).squeeze()

X = []
y = []

for i in range(NUM_USERS):
    index = np.where(task_index == i+1)
    min = index[0][0]
    max = index[0][-1] + 1
    X.append(X_Con[min:max])
    y.append(y_Con[min:max])


train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

for i in range(NUM_USERS):
    uname = i #'f_{0:05d}'.format(i)

    X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])

    train_data["user_data"][uname] = {'x': X_train.tolist(), 'y': y_train.tolist()}
    train_data['users'].append(uname)
    train_data['num_samples'].append(len(y_train))
    
    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test.tolist(), 'y': y_test.tolist()}
    test_data['num_samples'].append(len(y_test))

    
print("train", train_data['num_samples'])
print("test", test_data['num_samples'])
print("Num_samples:", train_data['num_samples']+ test_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
