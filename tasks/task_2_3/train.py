"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', default='actor-critic', type=str, choices=['actor-critic', 'reinforce'],
                        help='RL algorithm to use: "actor-critic" or "reinforce"')

    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=100, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.algorithm == 'actor-critic':
        from tasks.task_2_3.agent_actorCritic import Agent, Policy
    elif args.algorithm == 'reinforce':
        from tasks.task_2_3.agent_Reinforce import Agent, Policy
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:
            action, action_log_prob = agent.get_action(state)
            previous_state = state
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_log_prob, reward, done)
            train_reward += reward

            if args.algorithm == 'actor-critic':
                agent.update_policy()

        if args.algorithm == 'reinforce':
            agent.update_policy()

        if (episode + 1) % args.print_every == 0:
            print(f'Episode: {episode+1}/{args.n_episodes} | Return: {train_reward:.2f}')

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, f"{args.algorithm}.mdl")

    torch.save(agent.policy.state_dict(), save_path)
    print(f"\nTraining finished. Model saved to {save_path}")

if __name__ == '__main__':
    main()