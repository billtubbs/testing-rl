"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.

Advice on PPO tuning parameters:
https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
"""

import argparse
import numpy as np
import gym
from gym_CartPole_BT.envs.cartpole_bt_env import CartPoleBTEnv
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune import grid_search


# Parse command-line arguments
parser = argparse.ArgumentParser("Test run rllib model cartpole env")
parser.add_argument('--num_iters', type=int, default=1000, metavar='N',
                    help="Total number of training iterations")
parser.add_argument('--num_timesteps', type=int, default=10000, metavar='N',
                    help="Total number of timesteps to train for")
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='Initial seed for training')
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of parallel workers')
parser.add_argument('--num_gpus', type=int, default=0,
                    help='Number of GPUs to use')
parser.add_argument('--run', type=str, default="PPO",
                    help="Algorithm name (SAC, PPO, PG, A2C, A3C, IMPALA, "
                         "ES, DDPG, DQN, MARWIL, APEX, or APEX_DDPG)")
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate')
parser.add_argument('--lr_schedule', type=str, default=None,
                    help='Learning rate schedule')
parser.add_argument('--kl_target', type=float, default=0.01,
                    help='Target value for KL divergence (PPO)')
args = parser.parse_args()


def env_creator(env_config):
    """Return an env instance with given config"""
    return CartPoleBTEnv(
        goal_state=env_config['goal_state'],
        disturbances=env_config['disturbances'],
        initial_state=env_config['initial_state'],
        initial_state_variance=env_config['initial_state_variance']
    )

# Can also register the env creator function explicitly with:
register_env("CartPole-BT-v0", env_creator)
#trainer = ppo.PPOTrainer(env="CartPole-BT-v0")

config = {
    "env": "CartPole-BT-v0",
    "seed": args.seed,
    "num_gpus": 0,
    "num_workers": args.num_workers,
    "num_cpus_per_worker": 1,
    "env_config": {
                    'goal_state': [0.0, 0.0, np.pi, 0.0],
                    'disturbances': 'low',
                    'initial_state': 'goal',
                    'initial_state_variance': None
                },
    "lr": tune.grid_search((0.01, 0.001, 0.0001, 0.00001)),
    "eager": False
}

tune.run(
    "PPO",
    stop={
        "timesteps_total": args.num_timesteps,
    },
    config=config
)
