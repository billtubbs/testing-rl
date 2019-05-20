import os
import numpy as np
import gym
import gym_CartPole_BT

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

# I had a problem with my Mac OS X environment which required
# this fix (see here: https://github.com/dmlc/xgboost/issues/1715)
# Delete if you don't have this problem:
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env_name = 'CartPole-BT-dL-v0'
env = gym.make(env_name)
env = DummyVecEnv([lambda: env])

filename = f"ppo2_{env_name}"
if os.path.isfile(filename):
    model = PPO2.load(filename)
    model.set_env(env)
    print(f"Existing model loaded from file '{filename}'")
else:
    model = PPO2(MlpPolicy, env, verbose=1)

# Train model
model.learn(total_timesteps=50000, log_interval=10)
model.save(filename)
print(f"Model saved to file '{filename}'")

# Display animated runs with trained model
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