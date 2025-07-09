"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import sys
import os
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100000, type=int, help='Number of test episodes')
    parser.add_argument('--env', default="target", type=str, help='testing environment')

    return parser.parse_args()


args = parse_args()

'''
In the previous lines we have defined arguments to pass to the main. These arguments are then saved
in the global variable args such that are visible in every part of the program. Every argument has his 
default value, but you can specify every argument from the terminal when you throw the script. For example
since we have saved, through train.py, the weights of the REINFORCE network into the repo "models" when
you run the script you have to write from the terminal:
python test.py --model /home/ginevramuke/rl_mldl_25/models/REINFORCE.mdl --episodes 100
'''

def testing(model, env, n_episodes):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_episodes)
    return mean_reward, std_reward


def printMeanReward(label, mean, std):
    print(f"{label:<30} → Avg. Return: {mean:.2f} ± {std:.2f}")

def main():
    if args.env == "target":
        env = gym.make('CustomHopper-target-v0')
    elif args.env == "source":
        env = gym.make('CustomHopper-source-v0')
    else:
        exit()


    model = PPO.load(args.model) #load the model

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    # Evaluating
    results = testing(model, env, 50)

    # Printing
    printMeanReward("", results[0], results[1])

    if args.render:
        obs = env.reset()
        for _ in range(args.episodes):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()


if __name__ == '__main__':
    main()