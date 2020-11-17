import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import random
import json

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
    
X = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test)).squeeze()
task_index = np.concatenate((task_index_train, task_index_test)).squeeze()
argsort = np.argsort(task_index)
X = X[argsort]
y = np.array(y[argsort])
y = y-1
#y = tf.keras.utils.to_categorical(y, num_classes=NUM_LABELS)
task_index = task_index[argsort]
split_index = np.where(np.roll(task_index, 1) != task_index)[0][1:]
X = np.split(X, split_index)#.tolist()
y = np.split(y, split_index)#.tolist()

train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.75*num_samples)
    test_len = num_samples - train_len
    
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': X[i][:train_len].tolist(), 'y': y[i][:train_len].tolist()}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:].tolist(), 'y': y[i][train_len:].tolist()}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
