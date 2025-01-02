#!/bin/bash

# Basic NCCL settings
# export NCCL_DEBUG=INFO
# https://blog.csdn.net/qq_44091004/article/details/139445295
# 一般有些显卡30系列和40系列会存在不支持NCCL的某些功能，所以需要手动设置关掉
export NCCL_P2P_DISABLE=1
export NCCL_P2P_DISABLE=1 

# Launch without the no_python flag
accelerate launch --multi_gpu --num_processes=8 train.py -c cfgs/config.yaml