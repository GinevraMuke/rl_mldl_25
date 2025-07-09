import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *
import numpy as np
import argparse
from sb3_adapted_classes.policy import Policy
from sb3_adapted_classes.PPO import PPO_fine_tune
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
from torch import optim
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTerminal', default=False, action='store_true',
                        help='write --isTerminal if you are running this code from terminal instead of other script -> this allow you to store the configuration tried on wandb')
    parser.add_argument('--project_name', default= None, type=str, help='Name of the project on wandb')
    parser.add_argument('--env', default = "CustomHopper-target-v0", type = str, help='Name of the environment')
    parser.add_argument('--model', default="models_extension/ancient_deluge.mdl", type=str, help='MAML model path to be fine tuned with PPO')
    parser.add_argument('--model_destination_path', default = 'models_extension/MAML_PPO_fine_tuned', type = str, help='Destination path of the trained model')
    parser.add_argument('--save', default = False, action = 'store_true', help='call it if you want to save the final model')
    parser.add_argument(
        '--model_config',
        default="{'learning_rate' : 0.0009754622061100452, 'batch_size' : 7, 'num_steps' : 2539, 'num_epochs' : 21, 'gamma' : 0.99, 'clip_range' : 0.2979745482456337, 'vf_coef' : 0.5, 'ent_coef' : 0.0, 'gae_lambda' : 0.95, 'hidden_size' : 64}",
        type=str,
        help="Hyperparameters dictionary for PPO, it's important when throw the program from terminal to follow the default sintax"
    )
    return parser.parse_args()

args = parse_args()


'''
Create environment, train and save model
argparse.Namespace is needed to run train.py in other script (and not only from terminal),
for example for tuning of hyperparameters
'''

def train(args : argparse.Namespace, wandb_run = None):

    env = gym.make(args.env)
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]
    training_env = Monitor(env)
    # eval is a function that turns a string into a python code, in our case is needed
    # because we pass as argument to the script a string containing our configuration as a dictionary

    # PPO parameters
    config = eval(args.model_config)
    NUM_TIMESTEPS = 50000  # Data collected for support/query set.
    BATCH_SIZE = config['batch_size']  # mini-batch size to update PPO. MUST BE A DIVISOR OF NUM_TIMESTEPS
    NUM_EPOCHS = config['num_epochs']  # num of train epochs of PPO
    GAMMA = config['gamma']  # discount factor
    CLIP_RANGE = config['clip_range']
    VF_COEF = config['vf_coef']  # value function coef
    ENT_COEF = config['ent_coef']  # entropy coef
    GAE_LAMBDA = config['gae_lambda']
    HIDDEN_SIZE = config['hidden_size']
    LR = config['learning_rate']
    N_STEPS = config['num_steps']
    MODEL_SAVE_PATH = args.model_destination_path
    maml_policy = Policy(observation_space_dim, action_space_dim, hidden_dim=HIDDEN_SIZE)
    maml_policy.load_state_dict(torch.load(args.model, weights_only=True), strict=True)
    optimizer = optim.Adam(maml_policy.parameters(), lr=LR)
    model = PPO_fine_tune(env=training_env, policy = maml_policy, optimizer=optimizer,
                n_steps= N_STEPS,
                n_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                clip_range=CLIP_RANGE,
                ent_coef=ENT_COEF,
                vf_coef=VF_COEF,
                gae_lambda=GAE_LAMBDA,
                verbose=1,
                tensorboard_log= './ppo_hopper_tensorboard/'
                )

    run = wandb_run

    if args.project_name is not None and args.isTerminal:
        model_conf = eval(args.model_config)
        run = wandb.init(
            project=args.project_name,
            config=model_conf,
        )

    # Training and saving results
    if args.project_name is not None: #add callback for wandb
        model.learn(NUM_TIMESTEPS, callback= WandbCallback())
    else:
        model.learn(NUM_TIMESTEPS)

    if args.save:
        final_policy_state_dict = model.policy.state_dict()
        torch.save(final_policy_state_dict, MODEL_SAVE_PATH)

    if run is not None and args.isTerminal:
        run.finish()

    #need to log tuning with optuna
    mean_reward = -np.inf
    if model.ep_info_buffer:
        last_rewards = [ep_info['r'] for ep_info in model.ep_info_buffer]
        if last_rewards:
            mean_reward = np.mean(last_rewards)
    return mean_reward


def main():

    train(args)


if __name__ == '__main__':
    main()