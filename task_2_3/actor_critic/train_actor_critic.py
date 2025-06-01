"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import wandb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *

from task_2_3.actor_critic.agent_actorCritic import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():
    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    #WANDB configuration
    run = wandb.init(
        project="actor_critic",
        config = {
            "n_episodes": args.n_episodes,
        }
    )

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    # Training
	#TASK 2 and 3: interleave data and collection to policy updates
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    # Episodi
    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset() #reset the environment and observe the initial state

        while not done:
            action, action_log_prob = agent.get_action(state)
            previous_state = state
            state, reward, done, info = env.step(action.detach().cpu().numpy())
            agent.store_outcome(previous_state, state, action_log_prob, reward, done) #so agent.states[0] will contain the first state and agent.next_states[0] will contain the second state
            train_reward += reward
            agent.update_policy()
        #track reward progress on wandb for multiple trajectories
        run.log({"run on episode" : train_reward,
                 "episode" : episode
                 })

        if (episode + 1) % args.print_every == 0:
            print('Training episode:', episode)
            print('Episode return:', train_reward)

    torch.save(agent.policy.state_dict(), "/home/ginevramuke/rl_mldl_25/models/actor_critic.mdl") #basically this line is saving the parameters of the model (agent.policy.state_dict are the parameters)
    run.finish()




if __name__ == '__main__':
    main()