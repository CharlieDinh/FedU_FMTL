# Multitask Federated Learning
This repository implements all experiments in the paper the ** Multitask for Federated Learning **.
  
Authors: 

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc, doesn't require GPU.
  
# Dataset: We use 3 datasets:

Download Link: https://drive.google.com/drive/folders/1DZm7kQQqlDspwd4Q4nt6in6hvcMgtKs8?usp=sharing

- Google Glass (GLEAM) (38 tasks): This dataset consists of two hours of high resolution sensor data
collected from 38 participants wearing Google Glass for the purpose of activity recognition.
Following [41], we featurize the raw accelerometer, gyroscope, and magnetometer data into 180
statistical, spectral, and temporal features. We model each participant as a separate task, and
predict between eating and other activities (e.g., walking, talking, drinking).

- Human Activity Recognition (30 tasks): Mobile phone accelerometer and gyroscope data collected from
30 individuals, performing one of six activities: {walking, walking-upstairs, walking-downstairs,
sitting, standing, lying-down}. We use the provided 561-length feature vectors of time and
frequency domain variables generated for each instance [3]. We model each individual as a
separate task and predict between sitting and the other activities.

- Sensor (23 tasks): Acoustic, seismic, and infrared sensor data collected from a distributed network
of 23 sensors, deployed with the aim of classifying vehicles driving by a segment of road [13].
Each instance is described by 50 acoustic and 50 seismic features. We model each sensor as a
separate task and predict between AAV-type and DW-type vehicles

# Produce experiments and figures
- Human Activity
<pre><code>

python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.001 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1

python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1

python3 mocha.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Mocha --commet 0 --time 1 --gpu 0 --subusers 1

python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --times 20 --commet 0 --time 20 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 20 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 20 --gpu 0 --subusers  0.1
 </code></pre>

- Vehicle Sensor Activity
<pre><code>
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.001 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 1

python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 20 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 20 --gpu 0 --subusers 1

python3 mocha.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.001 --num_ƒ√global_iters 200  --algorithm Mocha --commet 0 --time 20 --gpu 0 --subusers 1

python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 20 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 20 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --times 20 --commet 0 --time 20 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 20 --gpu 0 --subusers 0.1
</code></pre>

- MNIST
<pre><code>
python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --num_global_iters 200  --algorithm FedAvg --times 10 --subusers 0.1 --commet 0 --gpu 0 --subusers 0.1
python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --subusers 0.1 --commet 0 --time 10 --gpu 1 --subusers 0.1
python3 main.py --dataset Mnist --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --subusers 0.1 --commet 0 --time 10 --gpu 1 --subusers 0.1
python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --subusers 0.1 --commet 0 --time 10 --gpu 1  --subusers 0.1
</code></pre>


- CIFAR10
<pre><code>
python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.001 --num_global_iters 1000  --algorithm FedAvg --times 1 --commet 0 --gpu 0 
python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.05 --L_k 0.01 --num_global_iters 1000  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.05 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.05 --beta 0.001  --num_global_iters 1000 --local_epochs 5 --algorithm PerAvg --commet 0 --time 5 --gpu 0  --subusers 0.1
</code></pre>