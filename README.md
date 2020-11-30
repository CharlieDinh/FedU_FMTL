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
