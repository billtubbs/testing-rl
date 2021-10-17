import gym
from gym_CartPole_BT.envs.cartpole_bt_env import CartPoleBTEnv

from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env


def env_creator(env_config):
    return CartPoleBTEnv(env_config)  # return an env instance

env_name = 'CartPole-BT-dL-v0'
register_env(env_name, env_creator)

env_config = {
    'description': "Basic cart-pendulum system with low random disturbance",
    'disturbances': 'low'
}

analysis = tune.run(
    PPOTrainer, 
    stop={"training_iteration": 100}, 
    config={"env": env_name, "env_config": env_config, "framework": "torch"}
)
