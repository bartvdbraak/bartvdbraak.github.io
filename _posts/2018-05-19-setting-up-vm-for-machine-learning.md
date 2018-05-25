---
layout: post
title: "Setting up a VM for Machine Learning"
description: "A step-by-step guide to setup a Virtual Machine for running Machine Learning SampleRNN with CUDA support."
thumb_image: "documentation/thumbnail-setup-vm.png"
tags: [SampleRNN, Neural Networks, Music, Machine Learning, Virtual Machine, CUDA]
---
## Introduction

This post will show a list of commands that will set up an environment for using [SampleRNN][0]. We will be using the [PyTorch][1] versions, since its easier to debug and setup.

The following architecture as Google Cloud VM instance:

| **MACHINE**: | 2 vCPUs, 7,5 GB |
| **CPU**: | Intel Sandy Bridge |
| **GPU**: | 1 NVIDIA Tesla K80 |
| **DISK**: | 50GB persistent |
| **OS**: | Ubuntu 16.04.4 LTS |
| **ARCH**: | x86_64|

The following packages/dependencies will be used:
- CUDA 9.0 (recommended)
- Anaconda 3.5.1
- Python 3.5
- PyTorch 0.3.1
- Librosa, Matplotlib, Natsort, Torch
- Tmux
- Comet_ml (optional)

## Connection

Generate a key with a username/password combination for secure connection through SSH.
~~~~~~~~
ssh-keygen -t rsa -f ~/.ssh/my-ssh-key -C <username>
~~~~~~~~
You will be prompted for a password that will be asked every SSH session: remember it.

Connect to your remote server (external IP address) with said credentials.
~~~~~~~~
ssh -i ~/.ssh/my-ssh-key <username>@<remote_ip>
~~~~~~~~

When your remote server's IP has changed:
~~~~~~~~
ssh-keygen -R <remote_ip>
~~~~~~~~


## General

Getting some updates and essential tools first for fresh installs of Ubuntu.

~~~~~~~~
sudo apt update
sudo apt upgrade
sudo apt-get install build-essential git
~~~~~~~~

## CUDA

For this sort of Machine Learning CUDA is recommended as it will speed up your training. If you do not have a Nvidia GPU or any GPU, I would suggest looking into the [free Google Cloud VM][2]'s (with free 300$ per Google Account) and [request a quotum for the Nvida K80 GPU][3].

Inside the remote machine:
~~~~~~~~
cd
mkdir CUDA
cd CUDA/
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb"
wget "http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb"
wget "http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb"
wget "http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl2_2.1.4-1+cuda9.0_amd64.deb"
wget "http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libnccl-dev_2.1.4-1+cuda9.0_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl2_2.1.4-1+cuda9.0_amd64.deb
sudo dpkg -i libnccl-dev_2.1.4-1+cuda9.0_amd64.deb
sudo apt-get update
sudo apt-get install cuda=9.0.176-1
sudo apt-get install libcudnn7-dev
sudo apt-get install libnccl-dev
~~~~~~~~
If you are prompted for the Terms of Service, just press `[space-bar]` till you're at the bottom of the page and type `accept` and `[enter]`. When any options require a yes/no, type `yes` and `[enter]`. When locations need to be specified just type `[enter]`

## Anaconda

Now we are going to set up our virtual environment manager: Anaconda. This will be easier when you are also keeping track of other versions of python for different projects. It also shows us if we are having trouble with dependency compatibility.
~~~~~~~~
cd
mkdir CONDA
cd CONDA/
wget "https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh"
sudo bash ./Anaconda3-5.1.0-Linux-x86_64.sh
~~~~~~~~
If you are prompted for the Terms of Service, just press `[space-bar]` till you're at the bottom of the page and type `accept` and `[enter]`. When any options require a yes/no, type `yes` and `[enter]`. When locations need to be specified just type `[enter]`

## GIT / DATASET (for SampleRNN)

We will be using the SampleRNN training as our project: specifically the [PyTorch][1] version.
~~~~~~~~
cd
mkdir CODE
cd CODE/
git clone https://github.com/deepsound-project/samplernn-pytorch.git
~~~~~~~~

Now we want to use their sample training data.

~~~~~~~~
sudo apt-get install youtube-dl
sudo apt-get install ffmpeg
cd samplernn-pytorch/datasets/
./download-from-youtube.sh "https://www.youtube.com/watch?v=EhO_MrRfftU" 8 piano
~~~~~~~~
There is a bug with youtube-dl which cause the DL-speed to cap at 50kbps and I am not too sure how to go around that. It will take some time in that case.

## Tmux sessions

We will be using `tmux` to keep our processes running for training and monitoring. If you close your SSH connection, the training won't be forced to quit.
~~~~~~~~
sudo apt-get install tmux
tmux new -s train
~~~~~~~~

Common commands:

| New session | `tmux new -s <name>` |
| Detach session | `[ctrl]+b d` |
| Attach session | `tmux a -t <name>` |
| List sessions | `tmux ls` |
| Kill session | `tmux kill-session -t <name>` |
| Kill all sessions | `tmux ls | grep : | cut -d. -f1 | awk '{print substr($1, 0, length($1)-1)}' | xargs kill` |

## Comet ML (optional)

For monitoring our training online, we can use Comet ML. You can get a free public account at [their website][4]. It will show you your API key in the example.

Install dependencies:
~~~~~~~~
source activate samplernn
conda install pip
pip install pyyaml
pip install cython
pip install comet_ml
~~~~~~~~

Change key in train.py:
~~~~~~~~
cd CODE/samplernn-pytorch/
nano python.py
~~~~~~~~
Navigate to this bit:
~~~~~~~~
'comet_key': None
~~~~~~~~
Change it to your Comet Projects API key:
~~~~~~~~
'comet_key': 'aaBbCCe1efFgghHiiJjkK2llM'
~~~~~~~~
Click `[ctrl]+O` to save with the same name, then `[ctrl]+X` to exit.

## Training

To start training, we now just activate our environment and start to run the code.

~~~~~~~~
source activate samplernn
cd CODE/samplernn-pytorch/
python train.py --exp TEST --frame_sizes 16 4 --n_rnn 2 --dataset piano
~~~~~~~~

### Happy training! ###

With these settings, every 1350 iterations (60-100 minutes) an epoch will end and a 5 second sequence will be generated in `results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:piano/samples`.

To send these to your own local computer, use the following commands:

`rsync -chavzP --stats -e 'ssh -i ~/.ssh/my-ssh-key' <username>@<remote_ip>:/home/<name>/CODE/samplernn-pytorch/results/exp:TEST-frame_sizes:16,4-n_rnn:2-dataset:piano/samples '<local folder>'`

### Questions ###

If you have any questions, even if you think they might be obvious to some, just ask them under this page or contact me privately!

[0]: https://github.com/soroushmehr/sampleRNN_ICLR2017 "SampleRNN"
[1]: https://github.com/deepsound-project/samplernn-pytorch "SampleRNN PyTorch"
[2]: https://cloud.google.com/compute/?hl=nl "Google Cloud's Compute Engine"
[3]: https://stackoverflow.com/questions/45227064/how-to-request-gpu-quota-increase-in-google-cloud#answer-49737435 "How to increase GPU Quota"
[4]: https://www.comet.ml "Comet ML"