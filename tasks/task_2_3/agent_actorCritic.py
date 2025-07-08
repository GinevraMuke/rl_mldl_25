import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma, baseline=0):
    '''
    Parameters:
        - r : vector containing the rewards obtained into a single trajectory
        - gamma : discount factor
    Returned:
        - discounted_r : vector of discounted reward for each timestamp t

    '''
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    if baseline > 0:
        return discounted_r - baseline
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)

        # Learned standard deviation for exploration at training time
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm, same of actor but

        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)

        """
            Critic
        """
        # TASK 3: forward in the critic network


        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        v_estimated = self.fc3_critic(x_critic)

        return normal_dist, v_estimated


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device  # the device in witch send our variables
        self.policy = policy.to(self.train_device)  # neural network to train
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)  # optimizer

        self.gamma = 0.99  # discount factor
        self.states = []  # states visited in a single trajectory
        self.next_states = []  #
        self.action_log_probs = []  # log-probability of each action chosen by the policy in a given state
        self.rewards = []  # rewards corresponding to states in the homonymous vector
        self.done = []

    def update_policy(self):
        '''
        - torch.stack: convert a list / array of tensors into a unique pytorch tensor
        - torch.to: send the variable to the assigned device
        - torch.squeeze: Returns a tensor with all specified dimensions of input of size 1 removed
        - torch.tensor: convert a single value into a tensor
        '''
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)
        '''
        In the previous lines store internally to the function local variables useful for policy update
        and convert them in the appropriate format.
        In the next line parameters of the agent are resettled to pass on another trajectory
        '''

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []

        #
        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        _, first_V = self.policy(states)
        _, second_V = self.policy(next_states)
        r = rewards[0]
        td = r + self.gamma*second_V - first_V

        '''
        update Critic
        '''
        critic_loss = td.pow(2)
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        '''
        update Actor
        '''
        actor_loss = -action_log_probs*td.detach()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()


        return

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)  # vector of three normal distributions

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:  # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob

    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

