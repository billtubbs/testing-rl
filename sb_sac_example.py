import os
import gym
import numpy as np
import gym_CartPole_BT

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

env_name = 'CartPole-BT-vH-v0'
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

filename = f"sac_{env_name}.pkl"
if os.path.isfile(filename):
    model = SAC.load(filename)
    model.set_env(env)
    print(f"Existing model loaded from file '{filename}'")
else:
    model = SAC(MlpPolicy, env, verbose=1)

model.learn(total_timesteps=50000, log_interval=10)
model.save(filename)
print(f"Model saved to file '{filename}'")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        s = input("Press enter to run again, 'q' to quit: ")
        if s.lower() == 'q':
            break
        obs = env.reset()

env.close()