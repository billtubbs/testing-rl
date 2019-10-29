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
from env_utils import simulation_rollout, simulation_rollouts
from control_baselines import LQR
from bayes_opt import BayesianOptimization


# Create logger based on config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test Bayesian optimization on gym environment.')
parser.add_argument('-e', '--env', type=str, default='CartPole-BT-vL-v0',
                    help="gym environment")
parser.add_argument('-l', "--log", help="log output to logfile.",
                    action="store_true")
parser.add_argument('-r', "--render", help="render animation",
                    action="store_true")
parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('-s', "--seed", help="seed for random number generator.",
                    type=int)
parser.add_argument('-n', '--n-repeats', type=int, default=5,
                    help="number of episodes (roll-outs) to average over.")
args = parser.parse_args()

# Initialize random number generator
rng = np.random.RandomState(args.seed)

# Create and initialize environment
logger.info(f"Initializing environment '{args.env}'...")
env = gym.make(args.env)
env.reset()  # TODO: Need to set the random seed

# Goal is to use Bayesian optimization to find the
# best K for the linear controller:
# u[t] = -Ky[t]

n_params = env.observation_space.shape[0]

# Parameters
logger.info(f"n_params: {n_params}")
logger.info(f"n_repeats: {args.n_repeats}")

# Initialize linear model
theta = np.zeros((1, n_params))
model = LQR(None, env, theta)

episode_count = 0
step_count = 0

# Bounded region of parameter space
pbounds = {
    'x': (-500, 500),
    'x_dot': (-500, 500), 
    'theta': (-2500, 2500), 
    'theta_dot': (-2500, 2500)
}

theta = np.zeros((1, n_params))
logging.info(f"Initial parameter values: {theta.__repr__()}")

def objective_function(x, x_dot, theta, theta_dot):

    global args, episode_count, step_count
    model.gain[:] = x, x_dot, theta, theta_dot
    states, actions, rewards = \
                simulation_rollouts(env, model, n_repeats=args.n_repeats, 
                                    log=False, render=False)
    episode_count += states.shape[0]
    step_count += states.shape[0] * states.shape[1]
    avg_cum_reward = rewards.sum(axis=1).mean()
    return avg_cum_reward

optimizer = BayesianOptimization(
    f=objective_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

start_time = datetime.datetime.now()
date_string = start_time.strftime('%Y-%m-%d')

# Prepare output data file
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
sub_dir = args.env
if not os.path.exists(os.path.join(results_dir, sub_dir)):
    os.makedirs(os.path.join(results_dir, sub_dir))
filename = f'bo-{date_string}.csv'
filepath = os.path.join(results_dir, sub_dir, filename)

def write_to_csv_file(episode_count, param_values, best_reward,
                      filename=filepath):
    with open(filepath, 'a') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow([episode_count] + param_values + [best_reward])

# Start with some random searches
optimizer.maximize(
    init_points=5,
    n_iter=0,
)

n_iter = 5
repeat_for = 1
done = False
while not done:

    for _ in range(repeat_for):
        optimizer.maximize(
            init_points=0,
            n_iter=n_iter,
        )

        best_params = optimizer.max['params']
        best_reward = optimizer.max['target']

        logger.info(f"episode_count: {episode_count}")
        logging.info(f"Best avg. reward: {best_reward}")
        logging.info(f"Best params: {best_params}")

        param_values = [best_params[name] for name in ('x', 'x_dot', 'theta', 'theta_dot')]
        write_to_csv_file(episode_count, param_values, best_reward)

    model.gain[:] = param_values

    # Note, this is a good solution:
    # theta = [-100.00,   -197.54,   1491.28,    668.44]

    while True:

        # Give user option to display rendered simulations
        s = input("Press enter to continue, 'r' to render, 'e' to edit, "
                    "and 'q' to quit: ").lower()
        
        try:
            d = int(s)
        except ValueError:
            d = None
        if s == 'q':
            done = True
            break
        elif s == '':
            break
        elif d is not None:
            repeat_for = d
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

# Close animation window
env.close()
