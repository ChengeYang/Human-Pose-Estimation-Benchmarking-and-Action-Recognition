# Install Python packages
$ pip3 install numpy
$ pip3 install pandas

# Install sklearn
$ pip3 install -U scikit-learn

# Install tensorflow
If CPU, its simply: $ pip install tensorflow  
If GPU, the steps are as follows:  

Install CUDA, cuDNN  

Install CUDA softwares, see:  
https://www.tensorflow.org/install/gpu  
search for "Ubuntu 18.04 (CUDA 10)" or "Ubuntu 16.04 (CUDA 10)", and copy and run the commands.  

However, I met with error on cuDNN, so I refer to this webpage for installing cuDNN: 
https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/  

Install tensorflow, see: https://www.tensorflow.org/install  
$ pip3 install tensorflow-gpu  

verification:  
$ python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"  

# Install for tf-pose-estimation  

For, go to the src/githubs directory and download the openpose github repo:  
$ cd src/githubs
$ git clone https://github.com/ildoonet/tf-pose-estimation  

Then, install all the required dependencies:   
$ cd tf-pose-estimation  
$ pip3 install -r requirements.txt  

$ cd tf_pose/pafprocess  
$ sudo apt install swig  
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace  

$ cd ../../  
$ cd models/graph/cmu  
$ bash download.sh  

$ pip3 install slidingwindow  
$ pip3 install opencv-python  
$ pip3 install opencv-contrib-python  
$ pip3 install simplejson  


# Test inference by tf-pose-estimation  
First go back to the root of tf-pose-estimation, then:  

Test a single image.  
$ python3 run.py --model=mobilenet_thin --resize=432x368 --image=./images/p1.jpg  

Realtime Webcam  
$ python3 run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0  

# Install keras for training action classification
$ sudo -H pip3 install keras
