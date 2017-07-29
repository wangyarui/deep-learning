# deep-learning
- This is my deep learning tour！ 
- This repository provides the learning abuout deeplearning. All those code come from deep learning tutorial and I made a simple change.
- This may take lot of time to run it.You have better configured Theano,Tensorflow with GPU.

## cnn

[Personal explanation](https://v.qq.com/x/page/m05215ungst.html) deep learning(1):CNN

> Usage

* git clone git@github.com:wangyarui/deep-learning.git
* cd cnn
* python conv.py
![conv](https://github.com/wangyarui/deep-learning/blob/master/cnn/figure_1.jpeg)
* python pooling 
![pooling](https://github.com/wangyarui/deep-learning/blob/master/cnn/figure_2.jpeg)
I took half hour using Theano with GPU,Please take patience.
* cd cnn_LeNet
* python convolutional_mlp.py
![final results](https://github.com/wangyarui/deep-learning/blob/master/cnn/results2.png)
![final results](https://github.com/wangyarui/deep-learning/blob/master/cnn/results1.png)

## rnn

[Personal explanation](https://v.qq.com/x/page/s0523iglx3p.html) deep learning(2):RNN

> Usage

* git clone git@github.com:wangyarui/deep-learning.git
* cd rnn
#### Install requirements

* pip install -r requirements.txt
### Install build tools

* sudo apt-get update
* sudo apt-get install -y build-essential git python-pip libfreetype6-dev libxft-dev libncurses-dev libopenblas-dev  gfortran python-matplotlib libblas-dev liblapack-dev libatlas-base-dev python-dev python-pydot linux-headers-generic linux-image-extra-virtual
* sudo pip install -U pip
### Install CUDA and Theano with GPU

### Clone the repo and install requirements

* git clone git@github.com:dennybritz/nn-theano.git
* cd nn-theano
* sudo pip install -r requirements.txt
### run main 

* python train-theano.py
* python rnn_theano.py

* I took three days using Theano with GPU,Please take more patience.

## AutoEncoder


> Usage

* cd AutoEncoder
* python basicAE.py
![results 1](https://github.com/wangyarui/deep-learning/blob/master/cnn/results1.png)
![results 2](https://github.com/wangyarui/deep-learning/blob/master/cnn/results1.png)
* python convDAE.py
![results 3](https://github.com/wangyarui/deep-learning/blob/master/cnn/results1.png)


