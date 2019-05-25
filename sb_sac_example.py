import os
import argparse
import numpy as np
import gym
import gym_CartPole_BT

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-dL-v0',
                    help="gym environment")
parser.add_argument('-s', "--show", help="display output",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
args = parser.parse_args()

# Create and initialize environment
if args.show:
    print(f"\nInitializing environment '{args.env}'...")
env = gym.make(args.env)
env = DummyVecEnv([lambda: env])

filename = f"sac_{args.env}.pkl"
if os.path.isfile(filename):
    model = SAC.load(filename)
    model.set_env(env)
    if args.show:
        print(f"Existing model loaded from file '{filename}'")
else:
    model = SAC(MlpPolicy, env, verbose=1)

# Train model
model.learn(total_timesteps=100, log_interval=10)
model.save(filename)
if args.show:
    print(f"Model saved to file '{filename}'")

if args.show:
    # Display animated runs with trained model
    obs = env.reset()
    cum_reward = 0.0
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        cum_reward += reward
        if args.render:
            env.render()
        if done:
            print(f"Reward: {round(cum_reward, 2)}")
            s = input("Press enter to run again, 'q' to quit: ")
            if s.lower() == 'q':
                break
            obs = env.reset()
            cum_reward = 0.0

env.close()
