import torch
import higher
import itertools
import gym
from env_extension.custom_hopper import *
import numpy as np
import os
from task_2_3.actor_critic.agent_actorCritic import Policy
import wandb # NUOVO: Importiamo la libreria Weights & Biases

# --- 1. SELEZIONE DEL DISPOSITIVO (GPU o CPU) ---
# Seleziona la GPU se disponibile, altrimenti usa la CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizzo del dispositivo: {device}")


def collect_trajectories(env, policy: Policy, num_steps: int):
    """
    Raccoglie un batch di esperienza in modo efficiente, pre-allocando i tensori.
    """
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    states_tensor = torch.zeros((num_steps, obs_dim), dtype=torch.float32, device=device)
    actions_tensor = torch.zeros((num_steps, action_dim), dtype=torch.float32, device=device)
    probs_tensor = torch.zeros(num_steps, dtype=torch.float32, device=device)
    vals_tensor = torch.zeros(num_steps, dtype=torch.float32, device=device)
    rewards_tensor = torch.zeros(num_steps, dtype=torch.float32, device=device)
    dones_tensor = torch.zeros(num_steps, dtype=torch.bool, device=device)

    obs = env.reset()

    for t in range(num_steps):
        state_tensor_input = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            dist, value = policy(state_tensor_input)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
        
        action_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, done, _ = env.step(action_np)

        states_tensor[t] = torch.tensor(obs, device=device)
        actions_tensor[t] = action.squeeze(0)
        probs_tensor[t] = log_prob.squeeze(0)
        vals_tensor[t] = value.squeeze()
        rewards_tensor[t] = torch.tensor(reward, device=device)
        dones_tensor[t] = torch.tensor(done, device=device)
        
        obs = next_obs
        if done:
            obs = env.reset()

    return {
        'states': states_tensor,
        'actions': actions_tensor,
        'probs': probs_tensor,
        'vals': vals_tensor,
        'rewards': rewards_tensor,
        'dones': dones_tensor
    }


def calculate_ppo_loss(
        trajectories: dict,
        actor_critic,
        n_epochs: int = 4, # Consiglio: aumentato per sfruttare meglio i dati
        batch_size: int = 64,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        gamma: float = 0.99
):
    """
    Calcola la loss totale di PPO, operando interamente su tensori GPU.
    """
    state_arr = trajectories['states']
    action_arr = trajectories['actions']
    old_prob_arr = trajectories['probs']
    vals_arr = trajectories['vals']
    reward_arr = trajectories['rewards']
    dones_arr = trajectories['dones']

    actor_losses = []
    critic_losses = []

    advantage = torch.zeros_like(reward_arr, device=device)
    last_advantage = 0.0
    for t in reversed(range(len(reward_arr))):
        mask = 1.0 - dones_arr[t].int()
        next_val = vals_arr[t + 1] if t < len(reward_arr) - 1 else 0.0
        delta = reward_arr[t] + gamma * next_val * mask - vals_arr[t]
        advantage[t] = delta + gamma * gae_lambda * mask * last_advantage
        last_advantage = advantage[t]
    
    # NUOVO: Normalizzazione dei vantaggi per stabilizzare l'addestramento
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    for _ in range(n_epochs):
        indices = torch.randperm(len(state_arr), device=device)
        batch_starts = torch.arange(0, len(state_arr), batch_size, device=device)
        
        for start in batch_starts:
            end = start + batch_size
            batch_indices = indices[start:end]

            states = state_arr[batch_indices]
            old_probs = old_prob_arr[batch_indices]
            actions = action_arr[batch_indices]

            dist, critic_value = actor_critic(states)
            critic_value = torch.squeeze(critic_value)

            new_probs = dist.log_prob(actions).sum(axis=-1)
            prob_ratio = torch.exp(new_probs - old_probs)
            
            weighted_probs = advantage[batch_indices] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage[batch_indices]
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

            returns = advantage[batch_indices] + vals_arr[batch_indices]
            critic_loss = ((returns - critic_value) ** 2).mean()

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

    total_loss = torch.stack(actor_losses).mean() + 0.5 * torch.stack(critic_losses).mean()
    return total_loss


# --- IPERPARAMETRI ---
hyperparameters = {
    "meta_lr": 1e-4, # Consiglio: ridotto per un apprendimento più stabile
    "inner_lr": 1e-2,
    "meta_batch_size": 10, # Consiglio: aumentato per un gradiente più stabile
    "inner_updates": 1,
    "num_meta_iterations": 500, # Aumentato per un addestramento più lungo
    "num_steps": 2048, # Aumentato per una stima del vantaggio più accurata
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "policy_clip": 0.2
}
SAVE_PATH = "trained_model"

# --- NUOVO: INIZIALIZZAZIONE DI WANDB ---
wandb.init(
    project="MAML-PPO-gemini-hyperparameters",
    config=hyperparameters
)

# --- SETUP ---
env = gym.make('CustomHopper-source-v0', udr_ranges={'thigh': 0.5, 'leg': 0.5, 'foot': 0.5})
obs_dim = env.observation_space.shape[-1]
action_dim = env.action_space.shape[-1]

meta_policy = Policy(obs_dim, action_dim)
meta_policy.to(device)

meta_optimizer = torch.optim.Adam(meta_policy.parameters(), lr=hyperparameters['meta_lr'])

# --- META-TRAINING LOOP ---
for meta_iter in range(hyperparameters['num_meta_iterations']):
    meta_losses = []
    # NUOVO: Liste per accumulare i reward da loggare
    support_rewards = []
    query_rewards = []

    for _ in range(hyperparameters['meta_batch_size']):
        env.set_random_parameters()
        
        inner_optimizer = torch.optim.SGD(meta_policy.parameters(), lr=hyperparameters['inner_lr'])
        with higher.innerloop_ctx(meta_policy, inner_optimizer, track_higher_grads=True) as (fast_policy, diff_optim):
            for _ in range(hyperparameters['inner_updates']):
                support_trajectories = collect_trajectories(env, fast_policy, num_steps=hyperparameters['num_steps'])
                inner_loss = calculate_ppo_loss(support_trajectories, fast_policy, n_epochs=hyperparameters['n_epochs'])
                diff_optim.step(inner_loss)

            query_trajectories = collect_trajectories(env, fast_policy, num_steps=hyperparameters['num_steps'])
            outer_loss = calculate_ppo_loss(query_trajectories, fast_policy, n_epochs=hyperparameters['n_epochs'])
            
            meta_losses.append(outer_loss)
            
            # NUOVO: Salviamo i reward medi di questo task
            support_rewards.append(support_trajectories['rewards'].mean().item())
            query_rewards.append(query_trajectories['rewards'].mean().item())

    meta_optimizer.zero_grad()
    total_meta_loss = torch.stack(meta_losses).mean()
    total_meta_loss.backward()
    meta_optimizer.step()

    # NUOVO: LOGGING SU WANDB
    wandb.log({
        "meta_loss": total_meta_loss.item(),
        "avg_outer_reward": np.mean(query_rewards), # Media dei reward post-adattamento
        "avg_pre_outer_reward": np.mean(support_rewards), # Media dei reward pre-adattamento
    }, step=meta_iter) # Usiamo meta_iter come asse x per i grafici

    if (meta_iter + 1) % 10 == 0:
        print(f"Meta-Iter {meta_iter + 1}/{hyperparameters['num_meta_iterations']}, "
              f"Meta-Loss: {total_meta_loss.item():.4f}, "
              f"Avg Query Reward: {np.mean(query_rewards):.2f}")

print("Addestramento completato.")

# --- SALVATAGGIO DEL MODELLO ---
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
model_save_path = os.path.join(SAVE_PATH, "meta_policy_final.mdl")
torch.save(meta_policy.state_dict(), model_save_path)
print(f"Modello addestrato salvato in: {model_save_path}")

# NUOVO: Chiudiamo il run di W&B
wandb.finish()