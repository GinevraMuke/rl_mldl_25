import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3 import SAC

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTerminal', default=False, action='store_true',
                        help='write --isTerminal if you are running this code from terminal instead of other script -> this allow you to store the configuration tried on wandb')
    parser.add_argument('--project_name', default= None, type=str, help='Name of the project on wandb')
    parser.add_argument('--env', default = "CustomHopper-source-v0", type = str, help='Name of the environment')
    parser.add_argument('--algo', default = "PPO", type = str, help='Name of the algorithm')
    parser.add_argument('--timesteps', default = 500000, type = int, help='Number of timesteps')
    parser.add_argument('--model_destination_path', default = 'models/DR', type = str, help='Destination path of the trained model')
    parser.add_argument('--save', default = False, action = 'store_true', help='call it if you want to save the final model')
    parser.add_argument(
        '--model_config',
        default="{'policy' : 'MlpPolicy', 'verbose' : 1 ,'learning_rate': 0.0009754622061100452, 'batch_size': 7, 'n_steps': 2539, 'n_epochs': 21, 'clip_range': 0.2979745482456337, 'tensorboard_log' : './ppo_hopper_tensorboard/'}",
        type=str,
        help="Hyperparameters dictionary for the model, it's important when throw the program from terminal to follow the default sintax"
    )
    parser.add_argument(
        '--env_config',
        default = "{'thigh': 0.5, 'leg': 0.5, 'foot': 0.5}",
        type = str,
        help = "environment configuration for Domain Randomization"
    )
    return parser.parse_args()

args = parse_args()

class DRCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages

    See documentation at https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        self.training_env.envs[0].set_random_parameters()
        '''
        training_env is an attribute of the custom class that represents our custom environment.
        The problem is that is a wrapper of multiple environments. Since we have only one environment,
        we select our environment using envs[0] and then selecting it's own method.
        '''

'''
Create environment, train and save model
argparse.Namespace is needed to run train.py in other script (and not only from terminal),
for example for tuning of hyperparameters
'''

def train(args : argparse.Namespace):

    training_env = Monitor(gym.make(args.env, udr_ranges = eval(args.env_config)))
    # eval is a function that turns a string into a python code, in our case is needed
    # because we pass as argument to the script a string containing our configuration as a dictionary

    print('State space:', training_env.observation_space)
    print('Action space:', training_env.action_space)
    print('Dynamics parameters:', training_env.get_parameters())

    model = createModel(args, training_env)

    run = None

    if args.project_name is not None and args.isTerminal:
        # evaluate configurations from string to dict
        model_conf = eval(args.model_config)
        env_conf = eval(args.env_config)

        combined_conf = {**model_conf, **env_conf}

        run = wandb.init(
            project=args.project_name,
            config=combined_conf,
            sync_tensorboard=True
        )

    # Training and saving results
    if args.project_name is not None: #add callback for wandb
        model.learn(args.timesteps, callback= CallbackList([WandbCallback(), DRCallback()]))
    else:
        model.learn(args.timesteps, callback=DRCallback())

    if args.save:
        model.save(args.model_destination_path)

    if run is not None:
        run.finish()

    return model


def createModel(args, env):
    if args.algo == 'PPO':
        model = PPO(
            env=env,
            **eval(args.model_config)
        )

    elif algorithm == 'SAC':
        # add parameters
        model = SAC(
            env=env,
            **eval(args.model_config)
        )

    else:
        raise ValueError("Choose 'PPO' or 'SAC'")

    return model


def main():

    train(args)


if __name__ == '__main__':
    main()