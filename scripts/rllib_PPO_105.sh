#!/bin/sh

# Shell script to launch one RL model testing experiment

python -u ../test_run_cartpole.py \
    --seed 105 \
    --num_timesteps 10000000 \
    --num_workers 46
