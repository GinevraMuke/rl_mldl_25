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
    parser.add_argument('--project_name', default= None, type=str, help='Name of the project on wandb')
    parser.add_argument('--env', default = "CustomHopper-source-v0", type = str, help='Name of the environment')
    parser.add_argument('--algo', default = "PPO", type = str, help='Name of the algorithm')
    parser.add_argument('--timesteps', default = 500000, type = int, help='Number of timesteps')
    parser.add_argument('--model_destination_path', default = 'models/PPO_source', type = str, help='Destination path of the trained model')
    parser.add_argument('--save', default = False, action = 'store_true', help='call it if you want to save the final model')
    parser.add_argument(
        '--model_config',
        default="{'policy' : 'MlpPolicy', 'verbose' : 1 ,'learning_rate': 0.001812633627871949, 'batch_size': 298, 'n_steps': 3859, 'n_epochs': 27, 'clip_range': 0.2864522385651961, 'ent_coef' :  0.004448319594844347, 'gae_lambda' : 0.9280365298599641, 'gamma' : 0.9925945837930582,'tensorboard_log' : './ppo_hopper_tensorboard/'}",
        type=str,
        help="Hyperparameters dictionary for the model, it's important when throw the program from terminal to follow the default sintax"
    )
    return parser.parse_args()

args = parse_args()


def train(args : argparse.Namespace):
    training_env = Monitor(gym.make(args.env))
    # eval is a function that turns a string into a python code, in our case is needed
    # because we pass as argument to the script a string containing our configuration as a dictionary

    print('State space:', training_env.observation_space)
    print('Action space:', training_env.action_space)
    print('Dynamics parameters:', training_env.get_parameters())

    model = createModel(args, training_env)

    #define wandb
    run = None
    if args.project_name is not None:
        run = wandb.init(
            project=args.project_name,
            config = eval(args.model_config),
            sync_tensorboard=True
        )

    # Training and saving results
    if args.project_name is not None: #add callback for wandb
        model.learn(args.timesteps, callback= WandbCallback())
    else:
        model.learn(args.timesteps)

    if run is not None:
        run.finish()

    if args.save:
        model.save(args.model_destination_path)

    return model


def createModel(args, env):
    if args.algo == 'PPO':
        model = PPO(
            env=env,
            **eval(args.model_config)
        )

    elif args.algo == 'SAC':
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