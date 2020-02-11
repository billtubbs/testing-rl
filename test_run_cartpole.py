# Based on example here:
# https://ray.readthedocs.io/en/latest/rllib.html

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import gym
from gym_CartPole_BT.envs.cartpole_bt_env import CartPoleBTEnv
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser("Test run rllib model cartpole env")
parser.add_argument('--n_iters', type=int, default=1000, metavar='N',
                    help="Total number of training iterations")
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='Initial seed for training')
args = parser.parse_args()

config = {
    "env": CartPoleBTEnv,
    "seed": args.seed,
    "num_gpus": 0,
    "num_workers": 47,
    "num_cpus_per_worker": 1
}

stop = {
    "training_iteration": args.n_iters
}

tune.run(PPOTrainer, config=config, stop=stop)
