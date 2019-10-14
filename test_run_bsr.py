import argparse
import logging
import logging.config
import numpy as np
import pandas as pd
import gym
import gym_CartPole_BT
from control_baselines import LQR, BasicRandomSearch
from env_utils import simulation_rollout, simulation_rollouts


# Create logger based on config file
logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test this gym environment.')
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
parser.add_argument('-n', '--n-repeats', type=int, default=3,
                    help="number of episodes (roll-outs) to average over.")
parser.add_argument('-t', '--n-samples', type=int, default=20,
                    help="number of directions sampled per iteration.")
parser.add_argument('-a', '--alpha', type=float, default=0.25,
                    help="step size.")
parser.add_argument('-z', '--noise_sd', type=float, default=3.0,
                    help="standard deviation of the exploration noise.")
args = parser.parse_args()

# Initialize random number generator
rng = np.random.RandomState(args.seed)

# Create and initialize environment
logger.info(f"Initializing environment '{args.env}'...")
env = gym.make(args.env)
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

# Do an initial random search (this is not part of standard BRS)
# NOTE: THIS IS NOT PART OF THE STANDARD BRS ALGORITH
n_search = args.n_samples*args.n_repeats
logger.info(f"Initial random search of {n_search} points in parameter-space...")
cum_rewards = []
params = []
for i in range(n_search):
    model.gain[:] = np.random.normal(scale=args.noise_sd*5, size=n_params)
    trajectory = simulation_rollout(env, model, n_steps=None, 
                                    #n_repeats=args.n_repeats, 
                                    log=False, render=False)
    episode_count += 1
    step_count += len(trajectory)
    rewards = np.array([sar[2] for sar in trajectory])
    params.append(model.gain.copy())
    cum_rewards.append(sum(rewards))
logger.info(np.array(cum_rewards).astype(int))
best = np.argmax(cum_rewards)
best_params = params[best]
theta = np.array(best_params)
#theta = np.zeros((1, n_params))
logging.info(f"Initial parameter values: {theta.__repr__()}")

heading1 = f"{'j':>3s} {'ep.':>5s} {'theta':>40s} {'Reward':>8s}"
heading2 = '-'*len(heading1)
if args.log:
    logger.info(heading1)
    logger.info(heading2)

j = 0
n_iter = 5
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

    # Print update to std. output
    if args.log:
        avg_cum_reward = 0.5*(cum_rewards['-'] + cum_rewards['+']).mean().round(2)
        logger.info(f"{j:3d} {episode_count:5d} {str(theta.round(2)):>40s} "
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
