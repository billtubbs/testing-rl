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
from ray.tune import grid_search

# Can also register the env creator function explicitly with:
# register_env("corridor", lambda config: SimpleCorridor(config))

tune.run(
    "PPO",
    stop={
        "timesteps_total": 10000,
    },
    config={
        "env": CartPoleBTEnv,  # or "env-name" if registered above
        "vf_share_layers": True,
        "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
        "num_workers": 47,  # parallelism
        "env_config": {
            # TODO: Specify env parameters
        },
    },
)
