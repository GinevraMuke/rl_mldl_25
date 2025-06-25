import torch as th
from torch import optim
from policy import Policy
from env.custom_hopper import *
from PPO import PPO
import wandb
import higher
from stable_baselines3.common.monitor import Monitor


def main():
    '''
    --- HYPERPARAMETERS ---
    '''

    # PPO parameters
    NUM_TIMESTEPS = 1024  # Data collected for support/query set.
    BATCH_SIZE = 64  # mini-batch size to update PPO. MUST BE A DIVISOR OF NUM_TIMESTEPS
    NUM_EPOCHS = 4  # num of train epochs of PPO
    GAMMA = 0.99  # discount factor
    CLIP_RANGE = 0.2
    VF_COEF = 0.5  # value function coef
    ENT_COEF = 0.01  # entropy coef
    GAE_LAMBDA = 0.95
    # MAML parameters
    INNER_LR = 3e-3  # learning rate for inner policies
    OUTER_LR = 1e-3  # learning rate for meta policy
    INNER_UPDATES = 10  # number of inner policy updates (the total number of updates is INNER_UPDATES*NUM_EPOCHS)
    NUM_TASK = 3  # number of tasks, for each task an inner policy is trained
    NUM_META_ITERATIONS = 500  # number of iterations for outer loop
    MODEL_SAVE_PATH = "meta_policy_hopper.mdl"


    #--- SETUP ---
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")
    env = gym.make('CustomHopper-source-v0', udr_ranges={'thigh': 0.1, 'leg': 0.1, 'foot': 0.1})
    meta_policy = Policy(env.observation_space.shape[-1], env.action_space.shape[-1]).to(device)
    meta_optimizer = optim.Adam(meta_policy.parameters(), lr=OUTER_LR)


    #--- WANDB SETUP ---
    run = wandb.init(
        project="MAML-PPO-sb3",
        config={
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "clip_range": CLIP_RANGE,
            "num_task": NUM_TASK,
            "num_meta_iteration": NUM_META_ITERATIONS,
            "inner_updates": INNER_UPDATES,
            "inner_lr": INNER_LR,
            "outer_lr": OUTER_LR,
            "vf_coef": VF_COEF,
            "ent_coef": ENT_COEF
        }
    )


    # --- META-TRAINING LOOP ---
    for meta_iter in range(NUM_META_ITERATIONS):

        rewards_after_adaptation = []
        meta_optimizer.zero_grad()

        for task_index in range(NUM_TASK):
            #at each task we restart the environment to do domain randomization and compatibility with Monitr of sb3
            env = gym.make('CustomHopper-source-v0', udr_ranges={'thigh': 0.1, 'leg': 0.1, 'foot': 0.1})
            env.set_random_parameters() #domain randomization
            env = Monitor(env)

            #inner optimizer
            inner_optimizer = optim.SGD(meta_policy.parameters(), lr=INNER_LR)

            with higher.innerloop_ctx(meta_policy, inner_optimizer, copy_initial_weights=False) as (fast_policy, diff_optim):
                model = PPO(env=env, policy=fast_policy, optimizer=diff_optim,
                            n_steps=NUM_TIMESTEPS,
                            n_epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            gamma=GAMMA,
                            clip_range=CLIP_RANGE,
                            ent_coef=ENT_COEF,
                            vf_coef=VF_COEF,
                            gae_lambda=GAE_LAMBDA)
                for i in range(INNER_UPDATES):
                    model.learn(total_timesteps=NUM_TIMESTEPS, query = False)

                model.learn(total_timesteps=NUM_TIMESTEPS, query = True)
                ep_info = env.get_episode_rewards()
                if ep_info:
                    rewards_after_adaptation.append(ep_info[-1])



        # --- WANDB LOGGING ---
        log_dict = {
            "meta_iteration": meta_iter
        }
        # add mean only if we have collected data
        if rewards_after_adaptation:
            log_dict["mean_reward_after_adaptation"] = np.mean(rewards_after_adaptation)

        wandb.log(log_dict, step=meta_iter)


        meta_optimizer.step()
        print(f"------------------------------------------step {meta_iter} completed --------------------------------------------")

    print("training completed")
    th.save(meta_policy.state_dict(), MODEL_SAVE_PATH)
    print(f"model saved in: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()