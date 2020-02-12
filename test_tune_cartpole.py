"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

import numpy as np
import gym
from gym_CartPole_BT.envs.cartpole_bt_env import CartPoleBTEnv
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.tune import grid_search


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
}

tune.run(
    "PPO",
    stop={
        "timesteps_total": 10000,
    },
    config=config,
    },
)
