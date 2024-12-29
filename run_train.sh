#!/bin/bash

# Basic NCCL settings
# export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export NCCL_P2P_DISABLE=1 

# Launch without the no_python flag
accelerate launch --multi_gpu --num_processes=8 train.py -c cfgs/config.yaml