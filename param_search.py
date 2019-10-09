import os
import csv
import argparse
import numpy as np
import pandas as pd
import gym
import gym_CartPole_BT
from control_baselines import LQR

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-v0',
                    help="gym environment")
parser.add_argument('-s', "--show", help="display output",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")
args = parser.parse_args()


def run_episode(env, model, render=True, show=True):

    obs = env.reset()

    if render:
        # Open graphics window and draw animation
        env.render()

    if show:
        print(f"{'k':>3s}  {'u':>5s} {'reward':>6s} "
               "{'cum_reward':>10s}")
        print("-"*28)

    # Keep track of the cumulative rewards
    cum_reward = 0.0

    # Run one episode
    done = False
    while not done:

        # Determine control input
        u, _ = model.predict(obs)

        # Run simulation one time-step
        obs, reward, done, info = env.step(u)

        if render:
            # Update the animation
            env.render()

        # Process the reward
        cum_reward += reward

        if show:
            # Print updates
            print(f"{env.time_step:3d}: {u[0]:5.1f} "
                  f"{reward:6.2f} {cum_reward:10.1f}")

    return cum_reward


# Create and initialize environment
if args.show: print(f"\nInitializing environment '{args.env}'...")
env = gym.make(args.env)
env.reset()

# Use random search to find the best linear controller:
# u[t] = -Ky[t]

# Search parameters
n_samples = 20  # Number of directions sampled per iteration
n_params = 4  # Number of parameters to search for

# Good solution is
# theta = [-100.00,   -197.54,   1491.28,    668.44]

range_data = {
    0: (-500, 500, 21),
    1: (-500, 500, 21),
    2: (-2500, 2500, 21),
    3: (-2500, 2500, 21),
}

mesh_shape = []
print(f"{'Param':>5} {'Min':>8s} {'Max':>8s} {'n':>5s}")
for i, v in range_data.items():
    print(f"{i:5d} {v[0]:8.2f} {v[1]:8.2f} {v[2]:5d}")
    mesh_shape.append(v[2])
n = np.array(mesh_shape).prod()
print(f"Running tests for {n} parameter combinations...")

results = np.full(mesh_shape, np.nan)
param_values = [np.linspace(v[0], v[1], v[2]) for i, v in range_data.items()]

# Initialize linear model
theta = np.zeros((1, n_params))
model = LQR(None, env, theta)

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
sub_dir = args.model
if not os.path.exists(os.path.join(results_dir, sub_dir)):
    os.makedirs(os.path.join(results_dir, sub_dir))
filename = 'param_sweep_results.csv'
filepath = os.path.join(results_dir, sub_dir, filename)

with open(filepath, 'w') as f:

    csv_writer = csv.writer(f, delimiter=',')

    headings = f"{'i':>5s} {'idx':>12s} {'theta':>32s} {'Reward':>8s}"
    headings += '\n' + '-'*len(headings)
    if args.show:
        print(headings)

    for i, idx in enumerate(np.ndindex(results.shape)):
        theta = [values[i] for i, values in zip(idx, param_values)]
        model.gain[:] = theta
        cum_reward = run_episode(env, model, render=False, show=False)
        results[idx] = cum_reward

        # Write results to file
        csv_writer.writerow(theta + [cum_reward])

        # Print update to std. output
        if args.show:
            print(f"{i:5d} {str(idx):>12s} "
                f"{str(np.round(theta, 2)):>32s} {cum_reward:8.1f}")

max_reward = results.max()
idx_max = np.unravel_index(np.argmax(results, axis=None), results.shape)
theta_best = [values[i] for i, values in zip(idx_max, param_values)]
print(f"Best reward was {max_reward:.4f} when theta = {theta_best}")
print("Finished")
