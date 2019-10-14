"""Utility functions for running OpenAI Gym simulations and
recording results.
"""

import logging
import logging.config
import numpy as np
import gym
import gym_CartPole_BT

# Create logger based on config file
logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def simulation_rollout(env, model, n_steps=None, log=True, render=False):

    observation = env.reset()

    if render:
        # Open graphics window and draw animation
        env.render()

    # Print to std. output (i.e. the screen) and log
    if log:
        logger.info(f"{'k':>3s}  {'u':>5s} {'reward':>6s} {'cum_reward':>10s}")
        logger.info("-"*28)

    # Keep track of the states and rewards
    trajectory = []
    cum_reward = 0

    # Run one rollout (episode)
    done = False
    while not done:

        # Get control input from model (policy)
        action, _ = model.predict(observation)

        # Run simulation one time-step
        observation, reward, done, info = env.step(action)

        if render:
            # Update the animation
            env.render()

        # Store the results
        trajectory.append((observation, action, reward))
        cum_reward += reward

        # Print updates
        if log:
            logger.info(f"{env.time_step:3d}: {action[0]:5.1f} {reward:6.2f} {cum_reward:10.1f}")
        
        if n_steps is not None:
            n_steps -= 1
            if n_steps == 0:
                break

    return trajectory


def simulation_rollouts(env, model, n_steps=None, n_repeats=1, log=True, render=False):

    assert len(env.observation_space.shape) == 1, \
        "Environment must have 1-dim observation space."
    assert len(env.action_space.shape) == 1, \
        "Environment must have 1-dim action space."

    states, actions, rewards = None, None, None
    for i in range(0, n_repeats):

        trajectory = simulation_rollout(env, model, log=log, render=render)

        if states is None:
            if n_steps is None:
                # If length of rollout was not specified set it to actual
                # episode length
                n_steps = len(trajectory)
            
            # Initialize empty arrays to store results
            states = np.full((n_repeats, n_steps, env.observation_space.shape[0]),
                             np.nan)
            actions = np.full((n_repeats, n_steps, env.action_space.shape[0]), 
                              np.nan)
            rewards = np.full((n_repeats, n_steps), np.nan)

        states[i] = np.stack([sar[0] for sar in trajectory])
        actions[i] = np.stack([sar[1] for sar in trajectory])
        rewards[i] = np.array([sar[2] for sar in trajectory])

    return states, actions, rewards
