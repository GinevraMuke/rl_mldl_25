"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='actor-critic', type=str, choices=['actor-critic', 'reinforce'])
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=10, type=int, help='Number of test episodes')

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


def main():

	if args.algorithm == 'actor-critic':
		from tasks.task_2_3.agent_actorCritic import Agent, Policy
	elif args.algorithm == 'reinforce':
		from tasks.task_2_3.agent_Reinforce import Agent, Policy

	#env = gym.make('CustomHopper-source-v0')
	env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True) #load the model from the path specified in args.model

	agent = Agent(policy, device=args.device)

	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward

		print(f"Episode: {episode} | Return: {test_reward}")


if __name__ == '__main__':
	main()