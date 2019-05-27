"""This module parses all command line arguments to main.py"""
import argparse
import numpy as np

parser = argparse.ArgumentParser('Testing RL Algorithms in Tensorflow')
parser.add_argument('--env', type=str, metavar='ENV-ID',
                    help="Gym environment id. E.g. 'CartPole-v1'")
parser.add_argument('--model', type=str, metavar='MODEL-ID',
                    help="RL algorithm name")
parser.add_argument('--name', type=str, default=None, metavar='LOG-NAME',
                    help="Meaningful label to describe this run. If None, a "
                    "name is created with the environment name, model name "
                    "and timestamp at start of training.")
parser.add_argument('--comment', type=str, default='', metavar='COMMENT',
                    help="user-defined text to describe this run or the "
                    "project or experiment that it relates to (default: '')")
parser.add_argument('--n_steps', type=int, default=50000, metavar='N',
                    help="Total number of samples to train on")
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='Initial seed for training')
parser.add_argument('--reset-timesteps', action='store_true', default=False,
                    help="Whether or not to reset the current timestep "
                    "number (used in logging).")
parser.add_argument('--log-interval', type=int, default=10, metavar='LOGINT',
                    help="Number of timesteps before logging (default: 10)")
parser.add_argument('--log_dir', type=str, default='./',
                    help="directory for outputting log files. (Default: './')")
parser.add_argument('--save-model', action='store_true', default=False,
                    help="save model after each epoch. If no log directory "
                         "is provided, saves to the current directory.")
parser.add_argument('--overwrite', action='store_true', default=False,
                    help="Without this argument, the script will raise a "
                         "FileExistsError if you try to write to an "
                         "existing log-dir output directory.")
parser.add_argument('--plots', action='store_true', default=False,
                    help="Automatically generate plots of learning curves.")

group1 = parser.add_argument_group('Model hyperparameters')
# TODO: Add model hyper-parameters here...
group1.add_argument('--model-args',type=str, default='{}',
                    help="A dictionary of extra arguments passed to the "
                    "model. (default: '{}')")
