# Based on example here:
# https://ray.readthedocs.io/en/latest/rllib-training.html#basic-python-api

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
import gym
from gym_CartPole_BT.envs.cartpole_bt_env import CartPoleBTEnv
import argparse


# Parse command-line arguments
parser = argparse.ArgumentParser("Test run rllib model cartpole env")
parser.add_argument('--n_iters', type=int, default=1000, metavar='N',
                    help="Total number of training iterations")
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='Initial seed for training')
args = parser.parse_args()

# Some stats
# tf-gpu 45/48 CPUs, 0/1 GPUs : timesteps / s = 2288000 / 1235.9 = 1851   
# tf-cpu 10 workers, 4 cpus each :  40000 / 27.38 = 1460
# tf-cpu 46 workers, 1 cpus each : 110400 / 62.35 = 1770
# tf-cpu 4 workers, 11 cpus each : 72000 / 62.96 = 1143
# tf-cpu 47 workers, 1 cpus each : 169200 / 94.51 = 1790
# tf-gpu 47 workers, 1 cpus each : 169200 / 94.40 = 1792

ray.init()

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 47
config["num_cpus_per_worker"] = 1
config["eager"] = False
config["seed"] = args.seed
trainer = ppo.PPOTrainer(config=config, env=CartPoleBTEnv)

# Can optionally call trainer.restore(path) to load a checkpoint.
for i in range(args.n_iters):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

ray.shutdown()
