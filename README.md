# Human Pose Estimation and Action Recognition
#### Deep Learning Project, Winter 2019, Northwestern University
#### Group members: Chenge Yang, Zhicheng Yu, Feiyu Chen
-----------------------------------------------------------------------------------------
## Demo
<p align = "center">
  <img src = "action_recognition.gif" height = "480px">
</p>

-----------------------------------------------------------------------------------------
## Introduction
This project contain two main parts:
### 1. Human Pose Estimation Benchmarking

### 2. Online Skeleton-Based Action Recognition
Single-frame realtime human action recognition based on OpenPose. The pipeline is as follows:
* Realtime human pose estimation via tf-pose-estimation
* Data preprocessing
* Action recognition with DNN using TensorFlow / Keras
-----------------------------------------------------------------------------------------
## Dependencies
* Python (my version: 3.6.7)
* pandas & numpy
* scikit-learn
* tensorflow
* keras
-----------------------------------------------------------------------------------------
## Implementation
### Data preprocessing
The output of OpenPose can be found at [OpenPose Demo - Output](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md). To transfer the original skeleton data into the input of our neural network, three preprocessing methods are used, which are implemented in [data_preprocessing.py]() :
1. Head reference: all joint positions are converted to the x-y coordinates relative to the head joint.
2. Pose to angle: the 18 joint positions are converted to 8 joint angles: left / right shoulder, left / right elbow, left / right hip, left / right knee.
3. Normalization: all joint positions are converted to the x-y coordinates relative to the skeleton bounding box.

The third approach is used, which gives the best result and robustness.
### DNN model
We built our DNN model refering to [Online-Realtime-Action-Recognition-based-on-OpenPose](https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose). The DNN model is implemented in [training.py]() using Keras and Tensorflow. The model consists of an input layer, an output layer and three hidden layers. The output layer uses softmax to conduct a 5-class classification.
### Training
* Copy your dataset (must be .csv file) into the same directory as [training.py]()
* Run the following command:
```
python3 training.py --dataset [dataset_filename]
```
-----------------------------------------------------------------------------------------
## Acknowledgement
* [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
* [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
* [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation)
* [Online-Realtime-Action-Recognition-based-on-OpenPose](https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose)
