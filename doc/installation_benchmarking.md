# Installation

## AlphaPose
The general version Installation of AlphaPose can be found [here](https://github.com/MVIG-SJTU/AlphaPose).

The version we use for this project is [PyTorch version](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch), see the installation [here](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch). As required at the second step, you may want to use
```
pip install -r requirements.txt
```
to install the dependencies it needs. However, you may run into some issues that not all the dependencies be installed successfully. Our advice is to try to install them line by line. The 9th line of **requirements.txt** - ntpath is included in **python os module**, so if you already have os, you don't need to reinstall it.

See more details about the usage & examples of AlphaPose [here](https://github.com/MVIG-SJTU/AlphaPose/blob/pytorch/doc/run.md). You can choose different hyperparameters to increse the accuracy while slowing down the inference speed and vice versa.

## Openpose
Check the general installation [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md). Make sure to download and install the prerequisites for your particular operating system following [prerequisites.md](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/prerequisites.md).

### Important Tips
1. For Ubuntu: Anaconda should not be installed on your system. Anaconda includes a Protobuf version that is incompatible with Caffe. Either you uninstall anaconda and install protobuf via apt-get, or you compile your own Caffe and link it to OpenPose. Additionally, the compatible version should be 3.6.1. Check the package version by
```
pip show protobuf
```
2. After you install **CMake GUI**: Assuming your CMake downloaded folder is in {CMAKE_FOLDER_PATH}, everytime these instructions mentions Cmake-gui, you will have to replace that line by {CMAKE_FOLDER_PATH}/bin/cmake-gui.

 <p align="center">
   <img src="../images/cmake-python.png" width="400">
 </p>

3. At OpenPose Configuration step, make sure the `BUILD_PYTHON` flag is set in Cmake-gui if you want to install Python API. However, it sometimes may not work. Go into your /build/python directory and try
```
sudo make install
```
to build **openpose** Python API. It will by default build its path to /usr/local/python, so you would add this to your environmental variables:
```
export PYTHONPATH=/usr/local/python/:$PYTHONPATH
```
If you are on Ubuntu or OSX, you can add this to your `.bashrc`. To verify if the API is successfully added to your path, check if there is any error when you import it in python:
```
import openpose as op
```
