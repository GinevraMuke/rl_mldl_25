"""Test a random policy on the OpenAI Gym Hopper environment.

    
    TASK 1: Play around with this code to get familiar with the
            Hopper environment.

            For example:
                - What is the state space in the Hopper environment? Is it discrete or continuous?
                Continuos
        		- What is the action space in the Hopper environment? Is it discrete or continuous?
        		 Continuos but in reality the range of action is always between -1 and 1, but these numbers are in float 32 so this
        		 range can be seen as continuos.
                - What is the mass value of each link of the Hopper environment, in the source and target variants respectively?
                - what happens if you don't reset the environment even after the episode is over?
                AssertionError: Cannot call env.step() before calling reset()
                - When exactly is the episode over?
                when done became true
                - What is an action here?
                it's a vector of 3 numbers between -1 and 1 that define the strength to apply to each part of the leg
"""
import pdb

import gym
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *


def main():
	env = gym.make('CustomHopper-source-v0')
	#env = gym.make('CustomHopper-target-v0')

	print('State space:', env.observation_space) # state-space
	print('Action space:', env.action_space) # action-space
	print('Dynamics parameters:', env.get_parameters()) # masses of each link of the Hopper

	n_episodes = 500
	render = True

	for episode in range(n_episodes):
		test_reward = 0
		done = False
		state = env.reset()	# Reset environment to initial state

		while not done:  # Until the episode is over

			action = env.action_space.sample()	# Sample random action

			state, reward, done, info = env.step(action)	# Step the simulator to the next timestep

			if render:
				env.render() #launch the graphic simulation

			test_reward += reward

		print(f"Episode: {episode} | Return: {test_reward}")

	

if __name__ == '__main__':
	main()