# FedU: A Unified Framework for Federated Multi-Task Learning with Laplacian Regularization
This repository implements all experiments in the paper the **FedU: A Unified Framework for Federated Multi-Task Learning with Laplacian Regularization**.
  
Authors: Canh T. Dinh, Tung T. Vu, Nguyen H. Tran, Minh N. Dao, Hongyu Zhang.
https://arxiv.org/pdf/2102.07148.pdf

# Software requirements:
- numpy, scipy, torch, Pillow, matplotlib, tqdm, pandas, h5py, comet_ml

- To download the dependencies: **pip3 install -r requirements.txt**

- The code can be run on any pc, doesn't require GPU.
  
# Datasets:

- Human Activity Recognition (30 clients)
- Vehicle Sensor (23 clients)
- MNIST (100 clients)
- CIFAR (20 clients): This dataset will be downloaded and generated automatically when runing algorithms.

Download Link: https://drive.google.com/drive/folders/1DZm7kQQqlDspwd4Q4nt6in6hvcMgtKs8?usp=sharing

All dataset after downloading must be stored at folder \data

# Produce experiments and figures

## Table comparison for Multi-Task Learning

                              | Dataset        | Algorithm |         Test accuracy        |
                              |----------------|-----------|---------------|--------------|
                              |                            | Convex        | Non Convex   |
                              |----------------|-----------|---------------|--------------|
                              | Human Activity | FedU      | 99.10 ± 0.18  | 99.21 ± 0.15 |
                              |                | MOCHA     | 98.79 ± 0.04  |              |
                              |                | Local     | 98.29 ± 0.01  | 98.34 ± 0.03 |
                              |                | Global    | 93.79 ± 0.27  | 94.58 ± 0.16 |
                              | Vehicle Sensor | FedU      | 91.16 ± 0.02  | 95.43 ± 0.09 |
                              |                | MOCHA     | 90.94 ± 0.05  |              |
                              |                | Local     | 88.16 ± 0.05  | 92.10 ± 0.06 |
                              |                | Global    | 80.21 ± 0.12  | 83.00 ± 0.11 |
                              |----------------|-----------|---------------|--------------|

    <pre><code>
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU  --time 10  --subusers 1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --time 10  --subusers 1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --time 10 --subusers 1
    python3 mocha.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 100 --K 1000 --num_global_iters 200  --algorithm Mocha --time 10 --subusers 1

    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU  --time 10  --subusers 1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --time 10  --subusers 1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --time 10 --subusers 1
    
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU  --time 10  --subusers 1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Local --time 10  --subusers 1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Global --time 10 --subusers 1
    python3 mocha.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 100 --K 200 --num_global_iters 200  --algorithm Mocha --time 10 --subusers 1

    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU  --time 10  --subusers 1
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Local --time 10  --subusers 1
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Global --time 10 --subusers 1
    </code></pre>
 
## Table comparison for Persionalized Federated Learning

                              | Algorithm  |            Test accuracy     |
                              |------------|---------------|--------------|
                              |            | MNIST         | CIFAR10      |
                              | FedU       | 97.84 ± 0.02  | 79.45 ± 0.02 |
                              | MOCHA      | 97.80 ± 0.02  |              |
                              | pFedMe     | 95.38 ± 0.09  | 78.70 ± 0.05 |
                              | Per-FedAvg | 91.77 ± 0.23  | 67.61 ± 0.03 |
                              | FedAvg     | 90.14 ± 0.61  | 41.01 ± 1.03 |

    <pre><code>
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --num_global_iters 200  --algorithm FedAvg --times 10 --subusers 0.1
    python3 mocha.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 100 --K 2000 --num_global_iters 200  --algorithm Mocha --time 10 --subusers 0.1
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --subusers 0.1  --time 10 --subusers 0.1
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe  --time 10 --subusers 0.1
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --subusers 0.1 --time 10  --subusers 0.1

    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.005 --num_global_iters 200  --algorithm FedAvg --times 10 --subusers 0.1
    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.05 --L_k 0.03 --num_global_iters 200  --algorithm FedU --time 10 --subusers 0.1
    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --time 10 --subusers 0.1 --K 5
    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.01 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --time 10 --subusers 0.1
    </code></pre>

## Effect of eta:

### Human Activity
- Convex
    <pre><code>
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.001 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1

    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 mocha.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 100 --K 1000 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    </code></pre>

- Non Convex
    <pre><code>
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.001 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1

    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    </code></pre>

### Vehicle Sensor 
- Convex
    <pre><code>
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.001 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1

    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 mocha.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 100 --K 200 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    </code></pre>

- Non-Convex
    <pre><code>
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.001 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 1 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1

    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    </code></pre>

### MNIST
- Convex
    <pre><code>
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.001 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.005 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.05 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1

    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 mocha.py --dataset Mnist --model mclr --learning_rate 0.05 --L_k 100 --K 1000 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 1
    </code></pre>

- NonConvex
    <pre><code>
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --L_k 0.001 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --L_k 0.005 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --L_k 0.05 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1

    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Local --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm Global --commet 0 --time 10 --gpu 0 --subusers 1 --cutoff 1
    </code></pre>


## Comparison for Persionalized Federated Learning using full data

### Human Activity
    <pre><code>
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset human_activity --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers  0.1
    python3 mocha.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 100 --K 1000 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 0.1 
    
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset human_activity --model dnn --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers  0.1
    </code></pre>

### Vehicle Sensor    
    <pre><code>
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset vehicle_sensor --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 mocha.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 100 --K 200 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 0.1 

    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
    python3 main.py --dataset vehicle_sensor --model dnn --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers 0.1
    </code></pre>


## Comparison for Persionalized Federated Learning using cut-off data

### Human Activity
    <pre><code>
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 0.05 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg  --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 main.py --dataset human_activity --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 10 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2 --K 5
    python3 main.py --dataset human_activity --model mclr --batch_size 20 --learning_rate 0.01 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers  0.1 --cutoff 2
    python3 mocha.py --dataset human_activity --model mclr --learning_rate 0.03 --L_k 100 --K 1000 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 0.1  --cutoff 2
    
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg  --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 main.py --dataset human_activity --model dnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2 --K 5
    python3 main.py --dataset human_activity --model dnn --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers  0.1 --cutoff 2
    </code></pre>

### Vehicle Sensor    
    <pre><code> 
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg  --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2 
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 main.py --dataset vehicle_sensor --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2 --K 5
    python3 main.py --dataset vehicle_sensor --model mclr --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 mocha.py --dataset vehicle_sensor --model mclr --learning_rate 0.05 --L_k 100 --K 200 --num_global_iters 200  --algorithm Mocha --commet 0 --time 10 --gpu 0 --subusers 0.1  --cutoff 2

    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.05 --L_k 0.01 --num_global_iters 200  --algorithm FedU --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2 --K 5
    python3 main.py --dataset vehicle_sensor --model dnn --learning_rate 0.02 --num_global_iters 200  --algorithm FedAvg  --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    python3 main.py --dataset vehicle_sensor --model dnn --batch_size 20 --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --commet 0 --time 10 --gpu 0 --subusers 0.1 --cutoff 2
    </code></pre>

### MNIST
    <pre><code>
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --num_global_iters 200  --algorithm FedAvg --times 10 --subusers 0.1 --cutoff 2
    python3 mocha.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 100 --K 2000 --num_global_iters 200  --algorithm Mocha --time 10 --subusers 0.1 --cutoff 2
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --subusers 0.1  --time 10 --subusers 0.1 --cutoff 2
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe  --time 10 --subusers 0.1 --cutoff 2 --K 5
    python3 main.py --dataset Mnist --model mclr --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --subusers 0.1 --time 10  --subusers 0.1 --cutoff 2
    
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --num_global_iters 200  --algorithm FedAvg --times 10 --subusers 0.1 --cutoff 2
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --L_k 0.01 --num_global_iters 200  --algorithm FedU --subusers 0.1  --time 10 --subusers 0.1 --cutoff 2
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe  --time 10 --subusers 0.1 --cutoff 2 --K 5
    python3 main.py --dataset Mnist --model dnn --learning_rate 0.03 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --subusers 0.1 --time 10  --subusers 0.1 --cutoff 2
    </code></pre>

### CIFAR-10
    <pre><code>
    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.005 --num_global_iters 200  --algorithm FedAvg --times 10 --subusers 0.1 --cutoff 2
    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.05 --L_k 0.03 --num_global_iters 200  --algorithm FedU --time 10 --subusers 0.1 --cutoff 2
    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.01 --personal_learning_rate 0.01 --beta 1 --L_k 15 --num_global_iters 200 --algorithm pFedMe --time 10 --subusers 0.1 --cutoff 2 --K 5
    python3 main.py --dataset Cifar10 --model cnn --learning_rate 0.01 --beta 0.001  --num_global_iters 200 --local_epochs 5 --algorithm PerAvg --time 10 --subusers 0.1 --cutoff 2
    </code></pre>
