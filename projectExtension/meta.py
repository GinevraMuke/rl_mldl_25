import torch
import higher
import itertools  # ## NUOVO: Importiamo itertools per unire i parametri
from env_extension.custom_hopper import *
import numpy as np
from task_2_3.actor_critic.agent_actorCritic import Policy


def collect_trajectories(env, policy : Policy, num_steps: int):
    """
    Raccoglie un batch di esperienza per ambienti con azioni DISCRETE,
    compatibile con la tua classe Agent.

    Args:
        env: L'ambiente di Reinforcement Learning (Gym-like).
        actor: La rete neurale dell'attore.
        critic: La rete neurale del critico.
        num_steps: Il numero totale di transizioni da raccogliere.

    Returns:
        Un dizionario che mappa stringhe a tensori PyTorch.
    """

    # Inizializzazione dei buffer di storage
    trajectory_data = {
        'states': [], 'actions': [], 'probs': [],
        'vals': [], 'rewards': [], 'dones': []
    }

    obs = env.reset()

    for _ in range(num_steps):
        # Usiamo torch.no_grad() per efficienza, dato che è una fase di inferenza
        with torch.no_grad():
            # Prepariamo l'osservazione per le reti
            state_tensor = torch.tensor(np.array([obs]), dtype=torch.float)

            # Replichiamo la logica di 'choose_action'
            dist, value = policy(state_tensor)
            action = dist.sample()

            probs = torch.squeeze(dist.log_prob(action).sum(axis=-1)).item()
            action = action.detach().cpu().numpy().squeeze()
            value = value.detach().cpu().item()

        # Interazione con l'ambiente
        next_obs, reward, done, _ = env.step(action)

        # Memorizziamo l'esperienza, come fa il tuo metodo 'remember'
        trajectory_data['states'].append(obs)
        trajectory_data['actions'].append(action)
        trajectory_data['probs'].append(probs)
        trajectory_data['vals'].append(value)
        trajectory_data['rewards'].append(reward)
        trajectory_data['dones'].append(done)



        obs = next_obs
        if done:
            obs = env.reset()


    # Conversione finale in tensori PyTorch
    # Usiamo il device del modello per coerenza
    for key, value_list in trajectory_data.items():
        # Creiamo tensori NumPy prima, poi li convertiamo in tensori PyTorch
        print(key)
        np_array = np.array(value_list)
        print(np_array.shape)
        if key in ['states', 'vals', 'rewards', 'dones']:
            dtype = torch.float
        else:  # actions, probs
            dtype = torch.long if key == 'actions' else torch.float

        trajectory_data[key] = torch.tensor(np_array, dtype=dtype)

    return trajectory_data


def calculate_ppo_loss(
        trajectories: dict,
        actor_critic,
        n_epochs: int = 10,
        batch_size: int = 64,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        gamma: float = 0.99
):
    """
    Calcola la loss totale di PPO basandosi sulla tua implementazione.

    Args:
        trajectories: Il dizionario di tensori restituito da collect_trajectories.
        actor: La rete neurale dell'attore corrente.
        critic: La rete neurale del critico corrente.
        n_epochs: Numero di epoche di ottimizzazione.
        batch_size: Dimensione dei mini-batch.
        ... altri iperparametri di PPO.

    Returns:
        Un tensore scalare che rappresenta la loss totale di PPO, pronto per la backward().
    """
    # Estraiamo i dati dal dizionario
    state_arr = trajectories['states']
    action_arr = trajectories['actions']
    old_prob_arr = trajectories['probs']
    vals_arr = trajectories['vals']
    reward_arr = trajectories['rewards']
    dones_arr = trajectories['dones']

    # Inizializziamo liste per accumulare le loss di ogni mini-batch
    actor_losses = []
    critic_losses = []

    # Loop delle epoche, come nel tuo metodo 'learn'
    for _ in range(n_epochs):

        # --- 1. Calcolo dei Vantaggi (GAE) ---
        # Questa logica è copiata direttamente dal tuo metodo 'learn'
        advantage = torch.zeros(len(reward_arr), dtype=torch.float32).cpu()
        for t in range(len(reward_arr) - 1):
            discount = 1.0
            a_t = 0.0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + gamma * vals_arr[k + 1] * (1 - int(dones_arr[k])) - vals_arr[k])
                discount *= gamma * gae_lambda
            advantage[t] = a_t

        # --- 2. Loop sui Mini-Batch ---
        n_states = len(state_arr)
        batch_start = torch.arange(0, n_states, batch_size)
        indices = torch.arange(n_states, dtype=torch.int64)
        np.random.shuffle(indices.cpu().numpy())  # Shuffle degli indici

        batches = [indices[i:i + batch_size] for i in batch_start]

        for batch in batches:
            # Selezioniamo i dati per il mini-batch
            states = state_arr[batch]
            old_probs = old_prob_arr[batch]
            actions = action_arr[batch]

            # Forward pass con le reti correnti (quelle del ciclo interno di MAML)
            dist, critic_value = actor_critic(states)
            critic_value = torch.squeeze(critic_value)

            # Calcolo della loss dell'attore
            new_probs = dist.log_prob(actions).sum(axis=-1)
            prob_ratio = torch.exp(new_probs - old_probs)

            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantage[batch]
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

            # Calcolo della loss del critico
            returns = advantage[batch] + vals_arr[batch]
            critic_loss = ((returns - critic_value) ** 2).mean()

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

    # Calcoliamo la media delle loss su tutti i mini-batch e tutte le epoche
    total_loss = torch.stack(actor_losses).mean() + 0.5 * torch.stack(critic_losses).mean()

    return total_loss



# --- IPERPARAMETRI ---
META_LR = 1e-3  # Learning rate del ciclo esterno
INNER_LR = 1e-2  # Learning rate del ciclo interno
META_BATCH_SIZE = 10  # Numero di task per meta-update
INNER_UPDATES = 1  # Numero di passi di gradiente nel ciclo interno
NUM_META_ITERATIONS = 500

# --- SETUP ---
env = gym.make('CustomHopper-source-v0', udr_ranges={'thigh': 0.5, 'leg': 0.5, 'foot': 0.5})

obs_dim = env.observation_space.shape[-1]
action_dim = env.action_space.shape[-1]

# Meta-policy (i nostri parametri theta)
meta_policy = Policy(obs_dim, action_dim)

# Meta-ottimizzatore per il ciclo esterno
meta_optimizer = torch.optim.Adam(meta_policy.parameters(), lr=META_LR)

# --- META-TRAINING LOOP ---
for meta_iter in range(NUM_META_ITERATIONS):

    # 1. Campiona un batch di task dalla distribuzione
    #tasks = env.sample_tasks(META_BATCH_SIZE)

    meta_losses = []

    # Per ogni task nel batch...
    for _ in range(META_BATCH_SIZE):
        env.set_random_parameters()

        # 2. CREA COPIA FUNZIONALE DELLA POLICY PER IL CICLO INTERNO
        # `higher` rende la rete "stateless" e traccia i gradienti
        # attraverso gli aggiornamenti del ciclo interno.
        inner_optimizer = torch.optim.SGD(meta_policy.parameters(), lr=INNER_LR)
        with higher.innerloop_ctx(meta_policy, inner_optimizer, track_higher_grads=True) as (fast_policy, diff_optim):

            # 3. --- CICLO INTERNO (ADATTAMENTO) ---
            for _ in range(INNER_UPDATES):
                # a. Raccogli dati (support set) con la policy corrente (fast_policy)
                support_trajectories = collect_trajectories(env, fast_policy, num_steps=2048)

                # b. Calcola la loss di RL sul support set
                inner_loss = calculate_ppo_loss(support_trajectories, fast_policy)  # Usiamo la stessa rete per actor e critic

                # c. Aggiorna i parametri della copia (fast_policy)
                # Questo aggiornamento viene tracciato da `higher`
                diff_optim.step(inner_loss)

            # 4. --- VALUTAZIONE PER IL META-UPDATE ---
            # a. Raccogli nuovi dati (query set) con la policy *adattata*
            query_trajectories = collect_trajectories(env, fast_policy, num_steps=2048)

            # b. Calcola la meta-loss sul query set
            # Il gradiente di questa loss si propagherà indietro fino ai
            # parametri originali `meta_policy` (theta).
            outer_loss = calculate_ppo_loss(query_trajectories, fast_policy, fast_policy)

            # Aggiungi la loss del task corrente alla lista
            meta_losses.append(outer_loss)

    # 5. --- CICLO ESTERNO (META-UPDATE) ---
    # a. Calcola la media delle meta-loss e il gradiente aggregato
    meta_optimizer.zero_grad()
    total_meta_loss = torch.stack(meta_losses).mean()

    # b. Propaga indietro i gradienti e aggiorna i meta-parametri (theta)
    total_meta_loss.backward()
    meta_optimizer.step()

    print(f"Meta-Iterazione {meta_iter + 1}/{NUM_META_ITERATIONS}, Meta-Loss: {total_meta_loss.item():.4f}")

print("Addestramento completato.")