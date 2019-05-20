import gym
import numpy as np
import gym_CartPole_BT

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

env_name = 'CartPole-BT-dL-v0'
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000, log_interval=10)
filename = f"ppo2_{env_name}"
model.save(filename)

#del model # remove to demonstrate saving and loading
#model = PPO2.load(filename)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()