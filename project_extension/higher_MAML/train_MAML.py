import torch as th
from torch import optim
from sb3_adapted_classes.policy import Policy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *
from sb3_adapted_classes.PPO import PPO
import wandb
import higher
from stable_baselines3.common.monitor import Monitor
import argparse
import optuna

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isTerminal', default=False, action='store_true',
                        help='write --isTerminal if you are running this code from terminal instead of other script -> this allow you to store the configuration tried on wandb')
    parser.add_argument('--project_name', default= None, type=str, help='Name of the project on wandb')
    parser.add_argument('--model_destination_path', default = 'models_extension/MAML_best_model', type = str, help='Destination path of the trained model')
    parser.add_argument('--save', default = False, action = 'store_true', help='call it if you want to save the final model')
    parser.add_argument(
        '--model_config',
        default="{'num_steps' : 512, 'batch_size' : 32, 'num_epochs' : 9, 'gamma' : 0.9760560734442784, 'clip_range' : 0.2, 'vf_coef' : 0.9934584599984284, 'ent_coef' : 0.0010548304264607628, 'gae_lambda' : 0.998284238789931, 'inner_lr' : 0.00041622597864896503, 'outer_lr' : 0.0004819827345091337, 'inner_updates' : 3, 'num_task' : 25, 'num_meta_iterations' : 500, 'dr_coef' : 0.44556624461778305, 'hidden_size' : 256}",
        type=str,
        help="Hyperparameters dictionary for the model, it's important when throw the program from terminal to follow the default sintax"
    )
    return parser.parse_args()

def train(args : argparse.Namespace, trial = None):
    '''
    --- HYPERPARAMETERS ---
    '''
    config = eval(args.model_config)

    # PPO parameters
    NUM_TIMESTEPS = config['num_steps']  # Data collected for support/query set.
    BATCH_SIZE = config['batch_size']  # mini-batch size to update PPO. MUST BE A DIVISOR OF NUM_TIMESTEPS
    NUM_EPOCHS = config['num_epochs']  # num of train epochs of PPO
    GAMMA = config['gamma']  # discount factor
    CLIP_RANGE = config['clip_range']
    VF_COEF = config['vf_coef']  # value function coef
    ENT_COEF = config['ent_coef']  # entropy coef
    GAE_LAMBDA = config['gae_lambda']
    # MAML parameters
    INNER_LR = config['inner_lr']  # learning rate for inner policies
    OUTER_LR = config['outer_lr']  # learning rate for meta policy
    INNER_UPDATES = config['inner_updates']  # number of inner policy updates (the total number of updates is INNER_UPDATES*NUM_EPOCHS)
    NUM_TASK = config['num_task']  # number of tasks, for each task an inner policy is trained
    NUM_META_ITERATIONS = config['num_meta_iterations']  # number of iterations for outer loop
    DR_COEF = config['dr_coef']
    HIDDEN_SIZE = config['hidden_size']
    MODEL_SAVE_PATH = args.model_destination_path


    #--- SETUP ---
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")
    env = gym.make('CustomHopper-source-v0', udr_ranges={'thigh': DR_COEF, 'leg': DR_COEF, 'foot': DR_COEF})
    meta_policy = Policy(env.observation_space.shape[-1], env.action_space.shape[-1], HIDDEN_SIZE).to(device)
    meta_optimizer = optim.Adam(meta_policy.parameters(), lr=OUTER_LR)


    #--- WANDB SETUP ---
    run = None
    if args.project_name is not None and args.isTerminal:
        run = wandb.init(
            project=args.project_name,
            config= config
        )

    last_reward = None
    # --- META-TRAINING LOOP ---
    for meta_iter in range(NUM_META_ITERATIONS):

        rewards_after_adaptation = []
        meta_optimizer.zero_grad()

        for task_index in range(NUM_TASK):
            #at each task we restart the environment to do domain randomization and compatibility with Monitr of sb3
            env = gym.make('CustomHopper-source-v0', udr_ranges={'thigh': DR_COEF, 'leg': DR_COEF, 'foot': DR_COEF})
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


        mean_reward = np.mean(rewards_after_adaptation)
        last_reward = mean_reward

        # --- WANDB LOGGING ---
        log_dict = {
            "meta_iteration": meta_iter
        }
        # add mean only if we have collected data
        if rewards_after_adaptation:
            log_dict["mean_reward_after_adaptation"] = mean_reward

        if wandb.run is not None:
            log_dict = {"meta_iteration": meta_iter, "mean_reward_after_adaptation": mean_reward}
            wandb.log(log_dict, step=meta_iter)

        # --- OPTUNA LOGGING IN CASE OF TUNING MODE ---
        if trial:
            trial.report(mean_reward, meta_iter)
            if trial.should_prune():
                # pruning for speed up
                print(f"Trial pruned at step {meta_iter}.")
                raise optuna.TrialPruned()

        if meta_iter % 100 == 0:
            if args.save:
                th.save(meta_policy.state_dict(), f"{args.model_destination_path}_{meta_iter}")
                print(f"model saved in: {args.model_destination_path}_{meta_iter}")


        meta_optimizer.step()
        print(f"------------------------------------------step {meta_iter} completed --------------------------------------------")

    print("training completed")
    if args.save:
        th.save(meta_policy.state_dict(), args.model_destination_path)
        print(f"model saved in: {args.model_destination_path}")
    return last_reward

if __name__ == '__main__':
    args = parse_args()
    train(args)