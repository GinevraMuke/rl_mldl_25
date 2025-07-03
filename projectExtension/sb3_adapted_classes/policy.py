import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor


class Policy(nn.Module):
    '''
    A classic network architecture with some more methods for ensure compatibility with sb3 classes
    '''
    def __init__(self, state_space, action_space, hidden_dim = 64):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden_dim

        # --- Architecture ---
        self.actor_net = nn.Sequential(
            nn.Linear(state_space, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(self.hidden, action_space)
        self.actor_log_std = nn.Linear(self.hidden, action_space)

        self.critic_net = nn.Sequential(
            nn.Linear(state_space, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh()
        )
        self.critic_value = nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        #orthogonal initialization (more stable for RL)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def _get_distribution_and_value(self, obs):
        """
        Metodo helper per non ripetere codice. Prende un'osservazione
        e restituisce la distribuzione e il valore del critico.
        """
        #Ensure correct type for PPO and super() class OnPolicyAlgo
        obs = obs.to(torch.float32)

        # actor forward
        hidden_actor = self.actor_net(obs)
        mean = self.actor_mean(hidden_actor)

        # sd calculus
        log_std = self.actor_log_std(hidden_actor)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        distribution = Normal(mean, std)

        # critic forward
        hidden_critic = self.critic_net(obs)
        value = self.critic_value(hidden_critic)

        return distribution, value

    def forward(self, obs):
        """
        Questo metodo non è più usato direttamente da __call__ ma può essere
        mantenuto per compatibilità o rimosso. Lo teniamo per ora.
        """
        return self._get_distribution_and_value(obs)

    def __call__(self, obs: torch.Tensor):
        """
        Usato durante collect_rollouts.
        Deve restituire (azioni, valori, log_prob).
        """
        distribution, value = self._get_distribution_and_value(obs)

        actions = distribution.sample()
        log_prob = distribution.log_prob(actions).sum(axis=-1)

        return actions, value.flatten(), log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Usato durante train.
        Deve restituire (valori, log_prob, entropia).
        """
        distribution, value = self._get_distribution_and_value(obs)

        log_prob = distribution.log_prob(actions).sum(axis=-1)
        entropy = distribution.entropy().sum(axis=-1)

        return value.flatten(), log_prob, entropy

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Usato alla fine di collect_rollouts per il bootstrap.
        """
        obs = obs.to(torch.float32)
        hidden_critic = self.critic_net(obs)
        value = self.critic_value(hidden_critic)
        return value

    def set_training_mode(self, mode: bool):
        self.train(mode)

    def reset_noise(self, *_):
        pass

    def obs_to_tensor(self, observation: np.ndarray) -> torch.Tensor:
        # convert a numpy observation to torch tensor (used for compatibility with sb3)
        device = next(self.parameters()).device
        return obs_as_tensor(observation, device)