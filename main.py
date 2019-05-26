import os
import time
import platform
import random
import subprocess
import numpy as np
import yaml
import gym

import tensorflow as tf
from tensorflow.python.client import device_lib

from parsers import parser

# Parse input arguments with parsers module
args = parser.parse_args()

if args.env.startswith('CartPole-BT'):
    import gym_CartPole_BT

if args.model == 'PPO2':
    from stable_baselines import PPO2
    from stable_baselines.common.policies import MlpPolicy
    MODEL_CLASS = PPO2
elif args.model == 'SAC':
    from stable_baselines import SAC
    from stable_baselines.common.policies import MlpPolicy
    MODEL_CLASS = SAC
elif args.model == 'DDPG':
    from stable_baselines import DDPG
    from stable_baselines.ddpg.policies import MlpPolicy
    from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec
    MODEL_CLASS = DDPG
else:
    raise ValueError(f"Model '{args.model}' not recognized.")

# Common packages
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

# Add additional parameters
args.start_time = time.strftime('%Y-%m-%d %H:%M:%S')
args.hostname = platform.node()

# Unique identifier for this simulation
args.id = time.strftime('%y%m%d%H%M%S') + args.hostname.replace('.', '_')

# For PyTorch use the following:
# args.has_cuda = torch.cuda.is_available()
# if args.has_cuda:
#     args.num_gpus = torch.cuda.device_count()
# else:
#     args.num_gpus = 0
# torch.manual_seed(args.seed)

# For Tensorflow use the following:
args.has_cuda = tf.test.is_gpu_available()
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
if args.has_cuda:
    args.num_gpus = len(get_available_gpus())
else:
    args.num_gpus = 0
tf.random.set_random_seed(args.seed)

# Just in case these are used somewhere:
random.seed(args.seed)
np.random.seed(args.seed)

# Record git hash, so we know what the code looked like when this ran
args.git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()

# Create log dir
if args.log_dir is not None:
    os.makedirs(args.log_dir, exist_ok=True)

    if args.overwrite is False and len(os.listdir(args.log_dir)) > 1:
        raise FileExistsError("There are existing files in '" + args.log_dir +
                              "'. Use --overwrite argument to over-write them.")

best_mean_reward, n_steps = -np.inf, 0

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps
    (see ACER or PPO2).
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(args.log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(f"Best mean reward: {best_mean_reward:.2f} - "
                  f"Last mean reward: {mean_reward:.2f}")

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Save the best model
                if args.save_model:
                    model_file_path = os.path.join(args.log_dir,
                                                   'best_model.pkl')
                    _locals['self'].save(model_file_path)
    n_steps += 1
    return True

def main():

    # Save argument values to yaml file
    args_file_path = os.path.join(args.log_dir, 'args.yaml')
    with open(args_file_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # Create and wrap the environment
    env = gym.make(args.env)
    env = Monitor(env, args.log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    # Add some param noise for exploration
    if args.model == 'DDPG':
        param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2,
                                             desired_action_stddev=0.2)
    else:
        param_noise = None

    model = MODEL_CLASS(MlpPolicy, env, param_noise=param_noise,
                        memory_limit=int(1e6), verbose=0)

    # Train the agent
    model.learn(total_timesteps=args.n_steps, callback=callback)

    # Save the final model
    if args.save_model:
        model_file_path = os.path.join(args.log_dir,
                                       'model.pkl')
        model.save(model_file_path)
        print("Best and final models saved in ", os.path.abspath(args.log_dir))

    if args.plots:
        raise NotImplementedError
        # TODO: utils.make_learning_curve_plots(args.log_dir)

if __name__ == '__main__':
    main()
