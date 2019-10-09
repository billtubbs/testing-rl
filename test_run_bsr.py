import argparse
import numpy as np
import pandas as pd
import gym
import gym_CartPole_BT
from control_baselines import LQR

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-dL-v0',
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

# Basic Random Search (BRS) parameters
# See Mania et al. 2018.
alpha = 0.25  # Step size
n_samples = 20  # Number of directions sampled per iteration
noise_sd = 3  # standard deviation of the exploration noise
n_params = 4  # Number of parameters to search for

# Initialize linear model
theta = np.zeros((1, n_params))
model = LQR(None, env, theta)
episode_count = 0

# Do an initial random search (this is not part of standard BRS)
print("Initial random search of parameter-space...")
cum_rewards = []
params = []
for i in range(n_samples):
    model.gain[:] = np.random.normal(scale=noise_sd*5, size=n_params)
    cum_reward = run_episode(env, model, render=False, show=False)
    episode_count += 1
    params.append(model.gain.copy())
    cum_rewards.append(cum_reward)
best = np.argmax(cum_rewards)
best_params = params[best]
theta = np.array(best_params)



headings = f"{'j':>3s} {'ep.':>5s} {'theta':>32s} {'Reward':>8s}"
headings += '\n' + '-'*len(headings)
if args.show:
    print(headings)

j = 0
n_iter = 10
done = False
while not done:

    # Sample from standard normal distribution
    delta_values = np.random.randn(n_samples*n_params).reshape((n_samples, n_params))
    cum_rewards = {'+': [], '-': []}
    for delta in delta_values:
        model.gain[:] = theta + delta*noise_sd
        cum_reward = run_episode(env, model, render=False, show=False)
        episode_count += 1
        cum_rewards['+'].append(cum_reward)
        model.gain[:] = theta - delta*noise_sd
        cum_reward = run_episode(env, model, render=False, show=False)
        episode_count += 1
        cum_rewards['-'].append(cum_reward)
    
    # Update model parameters
    cum_rewards = {k: np.array(v) for k, v in cum_rewards.items()}
    dr = cum_rewards['+'] - cum_rewards['-']
    dr_dtheta = dr.reshape((n_samples, 1)) * delta_values     
    theta = theta + alpha * dr_dtheta.sum(axis=0) / n_samples 

    # Print update to std. output
    if args.show:
        avg_cum_reward = 0.5*(cum_rewards['-'] + cum_rewards['+']).mean().round(2)
        print(f"{j:3d} {episode_count:5d} {str(theta.round(2)):>32s} "
              f"{avg_cum_reward:8.1f}")

    # Note, this is a good solution:
    # theta = [-100.00,   -197.54,   1491.28,    668.44]

    # Display rendered simulations
    j += 1
    if j % n_iter == 0:
        while True:
            s = input("Press enter to continue, 'r' to render, and "
                      "'q' to quit: ").lower()
            if s == 'q':
                done = True
                break
            if s == '':
                break
            elif s == 'r':
                cum_reward = run_episode(env, model, render=args.render,
                                         show=args.verbose)
                print(f"Reward: {round(cum_reward, 2)}")

# Close animation window
env.close()
