#!/bin/sh

# Shell script to launch one RL model testing experiment

python -u ../test_run_cartpole.py \
    --run SAC \
    --seed 1 \
    --num_timesteps 1000000 \
    --num_workers 46 \
    --lr_schedule [0.01,0.001,0.0001,0.00001]
