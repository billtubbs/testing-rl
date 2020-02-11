#!/bin/sh

# Shell script to launch one RL model testing experiment

python -u ../test_run_cartpole.py \
    --seed 103
    --n_iters 1000
