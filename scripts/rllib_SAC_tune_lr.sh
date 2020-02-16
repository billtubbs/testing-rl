#!/bin/sh

# Shell script to launch one RL model testing experiment

python -u ../SAC_tune_cartpole.py \
    --seed 1 \
    --num_timesteps 1000000 \
    --num_workers 44
