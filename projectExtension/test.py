"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse
import torch
from env.custom_hopper import *
from sb3_adapted_classes.policy import Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="models_extension/meta_policy_chatHP.mdl", type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100, type=int, help='Number of test episodes')

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


def get_action(state, policy):
    """ state -> action (3-d), action_log_densities """
    x = torch.from_numpy(state).float().to(args.device)

    normal_dist, _ = policy._get_distribution_and_value(x)  # vector of three normal distributions

    return normal_dist.mean, None


def main():
    #env = gym.make('CustomHopper-source-v0')
    env = gym.make('CustomHopper-target-v0')

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim, hidden_dim=64)
    policy.load_state_dict(torch.load(args.model), strict=True)  # load the model from the path specified in args.model


    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:

            action, _ = get_action(state, policy)

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward}")


if __name__ == '__main__':
    main()