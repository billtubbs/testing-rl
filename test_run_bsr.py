import argparse
import os
import logging
import logging.config
import yaml
import csv
import datetime
import numpy as np
import pandas as pd
import gym
import gym_CartPole_BT
from control_baselines import LQR, BasicRandomSearch
from env_utils import simulation_rollout, simulation_rollouts
from file_utils import make_new_filename, write_to_csv_file


# Create logger based on config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test an algorithm on a gym environment')
parser.add_argument('-a', '--alpha', type=float, default=0.25,
                    help="step size.")
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-vL-v0',
                    help="gym environment")
parser.add_argument('-f', "--file", help="log results to csv file",
                    action="store_true")
parser.add_argument('-l', "--log", help="log output to logfile",
                    action="store_true")
parser.add_argument('-m', '--max_iter', type=int, default=5,
                    help="Number of episodes (roll-outs) before stopping")
parser.add_argument('-n', '--n-repeats', type=int, default=5,
                    help="number of episodes (roll-outs) to average over")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
parser.add_argument('-s', "--seed", type=int, default=None,
                    help="seed for random number generator",)
parser.add_argument('-t', '--n-samples', type=int, default=20,
                    help="number of directions sampled per iteration")
parser.add_argument('-u', "--user", help="stop for user input",
                    action="store_true")
parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('-z', '--noise_sd', type=float, default=3.0,
                    help="standard deviation of the exploration noise")
args = parser.parse_args()

# Initialize random number generator
rng = np.random.RandomState(args.seed)

# Create and initialize environment
logger.info(f"Initializing environment '{args.env}'...")
env = gym.make(args.env)
if args.seed is not None:
    env.seed(args.seed)
env.reset()

# Use random search to find the best linear controller:
# u[t] = -Ky[t]

n_params = env.observation_space.shape[0]

# Basic Random Search (BRS) parameters
# See Mania et al. 2018.
logger.info(f"alpha: {args.alpha}")
logger.info(f"n_samples: {args.n_samples}")
logger.info(f"noise_sd: {args.noise_sd}")
logger.info(f"n_params: {n_params}")
logger.info(f"n_repeats: {args.n_repeats}")

# Initialize linear model
theta = np.zeros((1, n_params))
model = LQR(None, env, theta)

episode_count = 0
step_count = 0

if args.file:
    # Prepare output data file
    results_dir = 'results'
    sub_dir = args.env
    dir_path = os.path.join(results_dir, sub_dir)
    start_time = datetime.datetime.now()
    filename = make_new_filename('brs', dir_path=dir_path, start_time=start_time, 
                                 ext='csv')
    filepath = os.path.join(dir_path, filename)
    logger.info(f"Output file: {filepath}")


# Do an initial random search (this is not part of standard BRS)
# NOTE: THIS IS NOT PART OF THE STANDARD BRS ALGORITH
#n_search = args.n_samples*args.n_repeats
#logger.info(f"Initial random search of {n_search} points in parameter-space...")
#cum_rewards = []
#params = []
#for i in range(n_search):
#    model.gain[:] = np.random.normal(scale=args.noise_sd*5, size=n_params)
#    trajectory = simulation_rollout(env, model, n_steps=None, 
#                                    #n_repeats=args.n_repeats, 
#                                    log=False, render=False)
#    episode_count += 1
#    step_count += len(trajectory)
#    rewards = np.array([sar[2] for sar in trajectory])
#    params.append(model.gain.copy())
#    cum_rewards.append(sum(rewards))
#logger.info(np.array(cum_rewards).astype(int))
#best = np.argmax(cum_rewards)
#best_params = params[best]
#theta = np.array(best_params)


theta = np.zeros((1, n_params))
logging.info(f"Initial parameter values: {theta.__repr__()}")

if args.log:
    heading = f"{'i':>3s} {'ep.':>5s} {'theta':>40s} {'Reward':>8s}"
    logger.info(heading)
    logger.info('-'*len(heading))

iteration = 0
done = False
while not done:

    # Sample from standard normal distribution
    delta_values = rng.randn(args.n_samples * n_params)\
                   .reshape((args.n_samples, n_params))
    cum_rewards = {'+': [], '-': []}
    for delta in delta_values:
        model.gain[:] = theta + delta * args.noise_sd
        states, actions, rewards = \
            simulation_rollouts(env, model, n_repeats=args.n_repeats, 
                                log=False, render=False)
        episode_count += states.shape[0]
        step_count += states.shape[0] * states.shape[1]
        avg_cum_reward = rewards.sum(axis=1).mean()
        cum_rewards['+'].append(avg_cum_reward)
        model.gain[:] = theta - delta * args.noise_sd
        states, actions, rewards = \
            simulation_rollouts(env, model, n_repeats=args.n_repeats, 
                                 log=False, render=False)
        episode_count += states.shape[0]
        step_count += states.shape[0] * states.shape[1]
        avg_cum_reward = rewards.sum(axis=1).mean()
        cum_rewards['-'].append(avg_cum_reward)
    
    # Update model parameters
    cum_rewards = {k: np.array(v) for k, v in cum_rewards.items()}
    dr = cum_rewards['+'] - cum_rewards['-']
    dr_dtheta = dr.reshape((args.n_samples, 1)) * delta_values     
    theta = theta + args.alpha * dr_dtheta.sum(axis=0) / args.n_samples 
    param_values = theta[0, :]

    # Print update to std. output
    if args.log:
        avg_cum_reward = 0.5*(cum_rewards['-'] + cum_rewards['+']).mean().round(2)
        str_rep = str(param_values.round(2).tolist())
        logger.info(f"{iteration:3d} {episode_count:5d} {str_rep:>40s} "
                    f"{avg_cum_reward:8.1f}")

    if args.file:
        write_to_csv_file(filepath, episode_count, param_values, avg_cum_reward)

    # Note, this is a good solution:
    # theta = [-100.00,   -197.54,   1491.28,    668.44]

    iteration += 1

    if args.user and (iteration % args.max_iter) == 0:
        # Ask user what to do next
        while True:
            s = input("Press enter to continue, 'r' to render, 'e' to edit, "
                    "and 'q' to quit: ").lower()
            if s == 'q':
                done = True
                break
            if s == '':
                break
            elif s == 'r':
                trajectory = simulation_rollout(env, model, n_steps=None, 
                                                log=False, render=True)
                rewards = np.array([sar[2] for sar in trajectory])
                logger.info(f"Reward: {round(sum(rewards), 2)}")
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
                logger.info(f"theta: {theta.__repr__()}")
    elif iteration >= args.max_iter:
        done = True

# Close animation window
env.close()
