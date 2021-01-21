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
dir_path = os.path.join(dir_path, 'vehicle_sensor/Original_Data')

x = []
y = []
task_index = []

for root, dir, file_names in os.walk(dir_path):
    if 'acoustic' not in root and 'seismic' not in root:
        x_tmp = []
        for file_name in file_names:
            if 'feat' in file_name:
                dt_tmp = pd.read_csv(os.path.join(root, file_name),  sep=' ',
                                         skipinitialspace=True, header=None).values[:, :50]
                x_tmp.append(dt_tmp)
        if len(x_tmp) == 2:
            x_tmp = np.concatenate(x_tmp, axis=1)
            x.append(x_tmp)
            task_index.append(int(os.path.basename(root)[1:])*np.ones(x_tmp.shape[0]))
            y.append(int('aav' in os.path.basename(os.path.dirname(root)))*np.ones(x_tmp.shape[0]))

x = np.concatenate(x)
y = np.concatenate(y)
#y = tf.keras.utils.to_categorical(y, num_classes=2)
task_index = np.concatenate(task_index)
argsort = np.argsort(task_index)
x = x[argsort]
y = y[argsort]
task_index = task_index[argsort]
split_index = np.where(np.roll(task_index, 1) != task_index)[0][1:]
X = np.split(x, split_index)
y = np.split(y, split_index)

print(len(X))
train_path = './data/train/vehicle_train.json'
test_path = './data/test/vehicle_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

for i in range(len(X)):
    uname = i
    X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])

    train_data["user_data"][uname] = {'x': X_train.tolist(), 'y': y_train.tolist()}
    train_data['users'].append(uname)
    train_data['num_samples'].append(len(y_train))
    
    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test.tolist(), 'y': y_test.tolist()}
    test_data['num_samples'].append(len(y_test))

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
