from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from tqdm import trange
import numpy as np
import random
import json
import os

random.seed(1)
np.random.seed(1)
NUM_USERS = 1000
NUM_LABELS = 3
# Setup directory for train/test data
train_path = './data/train/mnist_train.json'
test_path = './data/test/mnist_test.json'
dir_path = os.path.dirname(train_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
dir_path = os.path.dirname(test_path)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

# Get MNIST data, normalize, and divide by level
mnist = fetch_mldata('MNIST original', data_home='./data')
mu = np.mean(mnist.data.astype(np.float32), 0)
sigma = np.std(mnist.data.astype(np.float32), 0)
mnist.data = (mnist.data.astype(np.float32) - mu)/(sigma+0.001)
mnist_data = []
for i in trange(10):
    idx = mnist.target==i
    mnist_data.append(mnist.data[idx])

print("\nNumb samples of each label:\n", [len(v) for v in mnist_data])

###### CREATE USER DATA SPLIT #######
# Assign 100 samples to each user
X = [[] for _ in range(NUM_USERS)]
y = [[] for _ in range(NUM_USERS)]
idx = np.zeros(10, dtype=np.int64)
for user in range(NUM_USERS):
    for j in range(NUM_LABELS):  # 3 labels for each users
        #l = (2*user+j)%10
        l = (user + j) % 10
        print("L:", l)
        X[user] += mnist_data[l][idx[l]:idx[l]+10].tolist()
        y[user] += (l*np.ones(10)).tolist()
        idx[l] += 10

print("IDX1:", idx)  # counting samples for each labels

# Assign remaining sample by power law
user = 0
props = np.random.lognormal(0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
props = np.array([[[len(v)-NUM_USERS]] for v in mnist_data]) * \
    props/np.sum(props, (1, 2), keepdims=True)
# print("here:",props/np.sum(props,(1,2), keepdims=True))
#props = np.array([[[len(v)-100]] for v in mnist_data]) * \
#    props/np.sum(props, (1, 2), keepdims=True)
#idx = 1000*np.ones(10, dtype=np.int64)
# print("here2:",props)
for user in trange(NUM_USERS):
    for j in range(NUM_LABELS):  # 4 labels for each users
        # l = (2*user+j)%10
        l = (user + j) % 10
        num_samples = int(props[l, user//int(NUM_USERS/10), j])
        numran1 = random.randint(1, 20)
        #numran2 = random.randint(1, 10)
        num_samples = (num_samples) + numran1 #+ 150
        #if(num_samples <= 30): 
        #    num_samples = num_samples * 2
        if idx[l] + num_samples < len(mnist_data[l]):
            X[user] += mnist_data[l][idx[l]:idx[l]+num_samples].tolist()
            y[user] += (l*np.ones(num_samples)).tolist()
            idx[l] += num_samples
            print("check len os user:", user, j,
                  "len data", len(X[user]), num_samples)

print("IDX2:", idx) # counting samples for each labels

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}

# Setup 5 users
# for i in trange(5, ncols=120):
# Setup 5 users
# for i in trange(5, ncols=120):
for i in range(NUM_USERS):
    uname = 'f_{0:05d}'.format(i)
    X_train, X_test, y_train, y_test = train_test_split(X[i], y[i], train_size=0.75, stratify=y[i])

    train_data["user_data"][uname] = {'x': X_train, 'y': y_train}
    train_data['users'].append(uname)
    train_data['num_samples'].append(len(y_train))
    
    test_data['users'].append(uname)
    test_data["user_data"][uname] = {'x': X_test, 'y': y_test}
    test_data['num_samples'].append(len(y_test))

print("Num_samples:", train_data['num_samples'])
print("Total_samples:",sum(train_data['num_samples'] + test_data['num_samples']))
    
with open(train_path,'w') as outfile:
    json.dump(train_data, outfile)
with open(test_path, 'w') as outfile:
    json.dump(test_data, outfile)

print("Finish Generating Samples")
