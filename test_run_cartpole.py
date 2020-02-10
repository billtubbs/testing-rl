# Based on example here:
# https://ray.readthedocs.io/en/latest/rllib.html

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray import tune
import gym
from gym_CartPole_BT.envs.cartpole_bt_env import CartPoleBTEnv


tune.run(PPOTrainer, config={"env": CartPoleBTEnv})
