# Multitask Federated Learning
This repository implements all experiments in the paper the ** FedU: A Unified Framework for Federated Multi-Task Learning with Laplacian Regularization **.
  
Authors: 

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib.

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc, doesn't require GPU.
  
# Dataset: We use 4 datasets:

Download Link: https://drive.google.com/drive/folders/1DZm7kQQqlDspwd4Q4nt6in6hvcMgtKs8?usp=sharing
- Human Activity Recognition (30 clients)
- Sensor (23 clients)
- MNIST (100 clients)
- CIFAR (20 clients)

# Produce experiments and figures

- Table comparison for Multi-Task Learning

- Table comparison for Persionalized Federated Learning

- Human Activity
Convex
<pre><code>
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.001 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1

python3 mocha.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 100 --K 1000 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --times 20 --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers  0.1
 </code></pre>

Non Convex
<pre><code>
python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.001 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1

python3 mocha.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 100 --K 1000 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model dnn --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --times 20 --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model dnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset human_activity --model dnn --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers  0.1
 </code></pre>

- Vehicle Sensor Activity
Convex
<pre><code>
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.001 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1

python3 mocha.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 100 --K 200 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --times 20 --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
</code></pre>

Non-Convex
<pre><code>
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.001 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 1 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1

python3 mocha.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 100 --K 200 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 1

python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm SSGD --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --times 20 --commet 0 --time 10 --gpu 0 --subusers 0.1
python3 main.py --dataset vehicle_sensor --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
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
