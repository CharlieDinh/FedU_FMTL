import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import random
import json
import os
from scipy.stats import skew, kurtosis
from librosa.feature import spectral_centroid, spectral_rolloff, delta
from librosa.onset import onset_strength

NUM_USERS = 38 
NUM_LABELS = 2
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(dir_path)
dir_path = os.path.join(dir_path, 'gleam/Original_Data/')

def generate_gleam_data():
    x = []
    y = []
    for root, dir, file_names in os.walk(dir_path):
        for file_name in file_names:
            if 'sensorData' in file_name and not file_name[0] == '.':
                x.append(pd.read_csv(os.path.join(root, file_name)))
            if 'annotate' in file_name and not file_name[0] == '.':
                y.append(pd.read_csv(os.path.join(root, file_name)))

    def crest_factor(a):
        return np.linalg.norm(a, ord=np.inf) / np.linalg.norm(a, ord=2)

    def delta_mean(a):
        return delta(a).mean()

    def delta_delta_mean(a):
        return delta(a, order=2).mean()

    def extract_feature(f, a):
        return [[f(s_s) for s_s in s] for s in a]

    def extract_x_and_y(x_i, y_i):
        time = x_i['Unix Time'].values
        sensor = x_i['Sensor'].values
        value1 = x_i['Value1'].values
        value2 = x_i['Value2'].values
        value3 = x_i['Value3'].values

        window_lenght = 60 * 1000
        time = time - time[0]
        time = np.array(time)

        masks_time = [np.logical_and(time >= i * window_lenght, time <= (i + 1) * window_lenght)
                      for i in range(int(time[-1] / window_lenght))]
        masks_sensor = [sensor == s for s in set(sensor) if 'Light' not in s]

        sens_value1 = [[value1[np.logical_and(m, s)] for s in masks_sensor] for m in masks_time]
        sens_value2 = [[value2[np.logical_and(m, s)] for s in masks_sensor] for m in masks_time]
        sens_value3 = [[value3[np.logical_and(m, s)] for s in masks_sensor] for m in masks_time]

        features = [np.mean, np.var, skew, kurtosis, crest_factor, spectral_centroid, onset_strength,
                    spectral_rolloff, delta_mean, delta_delta_mean]

        feat_1 = np.transpose(np.array([extract_feature(f, sens_value1) for f in features]), [1, 0, 2])
        feat_2 = np.transpose(np.array([extract_feature(f, sens_value2) for f in features]), [1, 0, 2])
        feat_3 = np.transpose(np.array([extract_feature(f, sens_value3) for f in features]), [1, 0, 2])

        feat_1 = np.reshape(feat_1, (feat_1.shape[0], -1))
        feat_2 = np.reshape(feat_2, (feat_1.shape[0], -1))
        feat_3 = np.reshape(feat_3, (feat_1.shape[0], -1))

        features = np.concatenate((feat_1, feat_2, feat_3), axis=1)

        labels_time = y_i['unix time'].values
        labels_time = np.array(labels_time - labels_time[0])
        activity = y_i['Activity'].values=='eat'
        status = y_i['Status']

        time_stamp_activity = labels_time[status == 'start']
        activity = list(activity[status == 'start'])
        masks_label = [np.logical_and(time >= time_stamp_activity[i], time < time_stamp_activity[i + 1])
                       for i, _ in enumerate(zip(time_stamp_activity, activity)) if i + 1 < len(time_stamp_activity)]
        masks_label.append(time >= time_stamp_activity[-1])
        activity_l = [np.where(m, a, None) for m, a in zip(masks_label, activity)]
        activity_lc = np.stack(activity_l)
        act = np.array([next(i for i in item if i is not None) for item in activity_lc.T])
        _, act_u = np.unique(act, return_inverse=True)
        labels = [np.argmax(np.bincount(act_u[m])) for m in masks_time]
        return features, labels

    x, y = zip(*[extract_x_and_y(x_i, y_i) for x_i, y_i in zip(x, y)])
    np.save(os.path.join(dir_path, 'x.npy'), x)
    np.save(os.path.join(dir_path, 'y.npy'), y)
    return x.tolist(), y.tolist()

if os.path.isfile(os.path.join(dir_path, 'x.npy')) and os.path.isfile(os.path.join(dir_path, 'y.npy')):
    X = np.load(os.path.join(dir_path, 'x.npy'),allow_pickle=True)
    y = np.load(os.path.join(dir_path, 'y.npy'),allow_pickle=True)
else:
    X, y = generate_gleam_data()

#y = [np.array(y_i) for y_i in y]
X = X.tolist()
y = y.tolist()


train_path = './data/train/glem_train.json'
test_path = './data/test/glem_test.json'

dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

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
    data = X[i][:train_len].tolist()
    print(type(data))
    train_data['users'].append(uname) 
    train_data['user_data'][uname] = {'x': data}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(uname)
    test_data['user_data'][uname] = {'x': X[i][train_len:].tolist()}
    test_data['num_samples'].append(test_len)

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
