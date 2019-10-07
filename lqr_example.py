import argparse
import gym
import gym_CartPole_BT
from control_baselines import LQRCartPend

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test the gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-dL-v0',
                    help="gym environment")
parser.add_argument('-s', '--show', type=bool, default=True,
                    action="store_true", help="display output")
parser.add_argument('-r', '--render', type=bool,
                    action="store_true", help="render animation")
args = parser.parse_args()

# Create and initialize environment
if args.show:
    print(f"\nInitializing environment '{args.env}'...")
env = gym.make(args.env)

model = LQRCartPend(None, env, verbose=1)

if args.show:
    # Display animated runs with model
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
