import argparse
import logging
import numpy as np
import pandas as pd
import gym
import gym_CartPole_BT
from control_baselines import LQR, BasicRandomSearch

# Parse any arguments provided at the command-line
parser = argparse.ArgumentParser(description='Test this gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-vL-v0',
                    help="gym environment")
parser.add_argument('-d', "--show", help="display output",
                    action="store_true")
parser.add_argument('-l', "--log", help="log output to logfile.",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('-s', '--seed', type=int, default=None,
                    help="Random to seed.  Initializes random number generator.")
parser.add_argument('-n', '--n-repeats', type=int, default=3,
                    help="Number of episodes (roll-outs) to average over.")
parser.add_argument('-t', '--n-samples', type=int, default=20,
                    help="Number of directions sampled per iteration.")
parser.add_argument('-a', '--alpha', type=float, default=0.25,
                    help="Step size.")
parser.add_argument('-z', '--noise_sd', type=float, default=3.0,
                    help="Standard deviation of the exploration noise.")
args = parser.parse_args()

logging.basicConfig(filename='logfile.txt', filemode='w', level=logging.DEBUG,
                    format='%(asctime)s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')


def run_episode(env, model, render=True, show=True):

    obs = env.reset()

    if render:
        # Open graphics window and draw animation
        env.render()

    if show:
        print_and_log(f"{'k':>3s}  {'u':>5s} {'reward':>6s} "
                "{'cum_reward':>10s}")
        print_and_log("-"*28)

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

        # Print updates
        if show:
            print_and_log(f"{env.time_step:3d}: {u[0]:5.1f} "
                        f"{reward:6.2f} {cum_reward:10.1f}")

    return cum_reward


def run_episodes(env, model, n_repeats=1, render=True, show=True):

    cum_rewards = []
    for _ in range(n_repeats):
        cum_reward = run_episode(env, model, render=render, show=show)
        cum_rewards.append(cum_reward)

    return np.array(cum_rewards).mean()


def print_and_log(message):
    if args.show: 
        print(message)
    if args.log:
        logging.info(message)


# Initialize random number generator
rng = np.random.RandomState(args.seed)

# Create and initialize environment
print_and_log(f"Initializing environment '{args.env}'...")
env = gym.make(args.env)
env.reset()

# Use random search to find the best linear controller:
# u[t] = -Ky[t]

n_params = env.observation_space.shape[0]

# Basic Random Search (BRS) parameters
# See Mania et al. 2018.
print_and_log(f"alpha: {args.alpha}")
print_and_log(f"n_samples: {args.n_samples}")
print_and_log(f"noise_sd: {args.noise_sd}")
print_and_log(f"n_params: {n_params}")
print_and_log(f"n_repeats: {args.n_repeats}")

# Initialize linear model
theta = np.zeros((1, n_params))
model = LQR(None, env, theta)
episode_count = 0

# # Do an initial random search (this is not part of standard BRS)
# # NOTE: THIS IS NOT PART OF THE STANDARD BRS ALGORITH
# print_and_log("Initial random search of parameter-space...")
# cum_rewards = []
# params = []
# for i in range(args.n_samples):
#     model.gain[:] = np.random.normal(scale=args.noise_sd*5, size=n_params)
#     cum_reward = run_episodes(env, model, n_repeats=args.n_repeats, 
#                               render=False, show=False)
#     episode_count += 1
#     params.append(model.gain.copy())
#     cum_rewards.append(cum_reward)
# best = np.argmax(cum_rewards)
# best_params = params[best]
# theta = np.array(best_params)
theta = np.zeros((1, n_params))

heading1 = f"{'j':>3s} {'ep.':>5s} {'theta':>40s} {'Reward':>8s}"
heading2 = '-'*len(heading1)
print_and_log(heading1)
print_and_log(heading2)

j = 0
n_iter = 5
done = False
while not done:

    # Sample from standard normal distribution
    delta_values = rng.randn(args.n_samples*n_params)\
                   .reshape((args.n_samples, n_params))
    cum_rewards = {'+': [], '-': []}
    for delta in delta_values:
        model.gain[:] = theta + delta*args.noise_sd
        cum_reward = run_episodes(env, model, n_repeats=args.n_repeats, 
                                  render=False, show=False)
        episode_count += 1
        cum_rewards['+'].append(cum_reward)
        model.gain[:] = theta - delta*args.noise_sd
        cum_reward = run_episodes(env, model, n_repeats=args.n_repeats, 
                                  render=False, show=False)
        episode_count += 1
        cum_rewards['-'].append(cum_reward)
    
    # Update model parameters
    cum_rewards = {k: np.array(v) for k, v in cum_rewards.items()}
    dr = cum_rewards['+'] - cum_rewards['-']
    dr_dtheta = dr.reshape((args.n_samples, 1)) * delta_values     
    theta = theta + args.alpha * dr_dtheta.sum(axis=0) / args.n_samples 

    # Print update to std. output
    if args.show:
        avg_cum_reward = 0.5*(cum_rewards['-'] + cum_rewards['+']).mean().round(2)
        print_and_log(f"{j:3d} {episode_count:5d} {str(theta.round(2)):>40s} "
                      f"{avg_cum_reward:8.1f}")

    # Note, this is a good solution:
    # theta = [-100.00,   -197.54,   1491.28,    668.44]

    # Display rendered simulations
    j += 1
    if j % n_iter == 0:
        while True:
            s = input("Press enter to continue, 'r' to render, 'e' to edit, "
                      "and 'q' to quit: ").lower()
            if s == 'q':
                done = True
                break
            if s == '':
                break
            elif s == 'r':
                cum_reward = run_episode(env, model, render=args.render,
                                         show=args.verbose)
                print_and_log(f"Reward: {round(cum_reward, 2)}")
            elif s == 'e':
                message = "Enter theta values seprated by commas: "
                while True:
                    try:
                        x = np.fromstring(input(message), sep=',').reshape((1, n_params))
                    except ValueError:
                        print("Not understood, try again.")
                    else:
                        theta[:] = x
                        break
                print_and_log(f"theta: {theta.__repr__()}")

# Close animation window
env.close()
