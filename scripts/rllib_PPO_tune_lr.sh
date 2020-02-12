#!/bin/sh

# Shell script to launch one RL model testing experiment

python -u ../test_tune_cartpole.py \
    --seed 1 \
    --num_timesteps 10000 \
    --num_workers 46 \
