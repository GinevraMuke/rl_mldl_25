import torch
import torch.nn.functional as F
from torch.distributions import Normal
from collections import OrderedDict

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        # --- Actor network ---
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space) + init_sigma)

        # --- Critic network ---
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, mean=0., std=0.1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, params=None):
        """
        Esegue un forward pass funzionale per MAML.
        Se `params` è None, usa i parametri standard del modulo.
        Altrimenti, usa i parametri forniti nel dizionario.
        """
        if params is None:
            # --- Forward pass standard ---
            # Actor
            x_actor = self.tanh(self.fc1_actor(x))
            x_actor = self.tanh(self.fc2_actor(x_actor))
            action_mean = self.fc3_actor_mean(x_actor)
            sigma = self.sigma_activation(self.sigma)
            # Critic
            x_critic = self.tanh(self.fc1_critic(x))
            x_critic = self.tanh(self.fc2_critic(x_critic))
            v_estimated = self.fc3_critic(x_critic)
        else:
            # --- Forward pass funzionale con i parametri forniti ---
            # Actor
            x_actor = self.tanh(F.linear(x, params['fc1_actor.weight'], params['fc1_actor.bias']))
            x_actor = self.tanh(F.linear(x_actor, params['fc2_actor.weight'], params['fc2_actor.bias']))
            action_mean = F.linear(x_actor, params['fc3_actor_mean.weight'], params['fc3_actor_mean.bias'])
            sigma = self.sigma_activation(params['sigma'])
            # Critic
            x_critic = self.tanh(F.linear(x, params['fc1_critic.weight'], params['fc1_critic.bias']))
            x_critic = self.tanh(F.linear(x_critic, params['fc2_critic.weight'], params['fc2_critic.bias']))
            v_estimated = F.linear(x_critic, params['fc3_critic.weight'], params['fc3_critic.bias'])

        normal_dist = Normal(action_mean, sigma)
        return normal_dist, v_estimated


class MAMLActorCritic:
    def __init__(self, policy, device='cpu', inner_lr=1e-2, num_inner_steps=1, gamma=0.99, critic_loss_coeff=0.5):
        self.train_device = device
        self.meta_policy = policy.to(self.train_device)
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.gamma = gamma
        self.critic_loss_coeff = critic_loss_coeff

    def inner_loop_adapt(self, task_data):
        """
        Esegue l'inner loop di MAML per un singolo task usando Actor-Critic e restituisce
        i parametri adattati (differenziabili).

        Args:
            task_data: Un dizionario contenente tensori per 'states', 'actions', 'rewards',
                       'next_states', e 'dones'.

        Returns:
            OrderedDict: Un dizionario contenente i parametri adattati ('theta_prime').
        """
        # Estrai e prepara i dati del task
        states = task_data['states'].to(self.train_device)
        actions = task_data['actions'].to(self.train_device)
        rewards = task_data['rewards'].to(self.train_device).squeeze(-1)
        next_states = task_data['next_states'].to(self.train_device)
        dones = task_data['dones'].to(self.train_device).squeeze(-1)

        # 1. Copia i parametri della meta-policy in un dizionario. Questo è il nostro `theta`.
        #    Questo dizionario sarà aggiornato in modo funzionale.
        adapted_params = OrderedDict(self.meta_policy.named_parameters())

        for _ in range(self.num_inner_steps):
            # 2. Calcola i valori e le distribuzioni usando i parametri CORRENTI (adapted_params)
            #    Questo è il forward pass funzionale.
            _, v_current = self.meta_policy(states, params=adapted_params)
            _, v_next = self.meta_policy(next_states, params=adapted_params)

            # Stacca il gradiente dal valore dello stato successivo per la stabilità (target)
            v_next = v_next.detach()

            # 3. Calcola l'Advantage (usando TD error)
            target = rewards + self.gamma * v_next.squeeze(-1) * (1.0 - dones)
            advantages = (target - v_current.squeeze(
                -1)).detach()  # Tratta l'advantage come costante per la loss dell'attore

            # 4. Calcola la loss del Critico (MSE)
            critic_loss = F.mse_loss(v_current.squeeze(-1), target)

            # 5. Calcola la loss dell'Attore
            dist_current, _ = self.meta_policy(states, params=adapted_params)
            log_probs = dist_current.log_prob(actions).sum(dim=-1)
            actor_loss = -torch.mean(log_probs * advantages)

            # 6. Combina le loss e calcola i gradienti
            inner_loss = actor_loss + self.critic_loss_coeff * critic_loss

            # Calcola i gradienti rispetto ai parametri. `create_graph=True` è essenziale.
            grads = torch.autograd.grad(inner_loss, adapted_params.values(), create_graph=True)

            # 7. Esegui il passo di gradiente manualmente per ottenere i nuovi `adapted_params`
            adapted_params = OrderedDict(
                (name, param - self.inner_lr * grad)
                for ((name, param), grad) in zip(adapted_params.items(), grads)
            )

        return adapted_params

    def get_action(self, state, params=None, evaluation=False):
        """
        Ottiene un'azione usando un set di parametri specifico.

        Args:
            state (np.array): Lo stato corrente.
            params (dict, optional): Dizionario di parametri da usare. Se None, usa
                                     i parametri della meta-policy.
            evaluation (bool): Se True, restituisce la media invece di campionare.
        """
        x = torch.from_numpy(state).float().to(self.train_device).unsqueeze(0)  # Aggiungi batch dim

        # Esegui il forward pass (funzionale se `params` è fornito)
        dist, value = self.meta_policy(x, params=params)

        if evaluation:
            action = dist.mean
        else:
            action = dist.sample()

        action_log_prob = dist.log_prob(action).sum()

        # Rimuovi batch dim per restituire valori singoli
        return action.squeeze(0), action_log_prob, value.squeeze(0)
