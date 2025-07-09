import optuna
import wandb
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='PPO', type=str, choices=['PPO', 'SAC'],
                        help='The RL algorithm to tune (PPO or SAC).')

    parser.add_argument('--project_name', default=None, type=str,
                        help='Name of the project on Weights & Biases. If None, a default name will be generated.')

    parser.add_argument('--timesteps', default=50000, type=int,
                        help='Number of timesteps for training each trial.')

    parser.add_argument('--n_trials', default=50, type=int,
                        help='Number of Optuna trials to run.')

    return parser.parse_args()


def optimize_agent(trial, args):

    if args.algo == 'PPO':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_int('n_steps', 256, 4096),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'n_epochs': trial.suggest_int('n_epochs', 5, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 1.0),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.01),
        }
    elif args.algo == 'SAC':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
            'buffer_size': trial.suggest_int('buffer_size', 100_000, 1_000_000),
            'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
            'tau': trial.suggest_float('tau', 0.001, 0.02),
            'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8, 16]),  # Frequenza in step o episodi
            'gradient_steps': trial.suggest_int('gradient_steps', 1, 10),
        }
    else:
        raise ValueError(f"Algorithm {args.algo} not supported.")

    project_name = args.project_name if args.project_name else f"hopper-tuning-{args.algo}"
    run_name = f"{args.algo}-trial-{trial.number}"

    run = wandb.init(
        project=project_name,
        name=run_name,
        config=params,
        sync_tensorboard=True,
        reinit=True,
    )
    env = Monitor(gym.make('CustomHopper-source-v0'))
    tensorboard_log_path = f"./{args.algo}_tensorboard_logs/"

    if args.algo == 'PPO':
        model = PPO('MlpPolicy', env, tensorboard_log=tensorboard_log_path, verbose = 1, **params)
    elif args.algo == 'SAC':
        model = SAC('MlpPolicy', env, tensorboard_log=tensorboard_log_path, verbose = 1, **params)
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
            )
        )
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)

        wandb.log({"mean_reward_eval": mean_reward})

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        mean_reward = -float('inf')
    finally:
        run.finish()

    return mean_reward


def main():
    args = parse_args()

    study = optuna.create_study(direction='maximize')

    objective_func = lambda trial: optimize_agent(trial, args)

    study.optimize(objective_func, n_trials=args.n_trials)

    print(f"\nOptimization finished for algorithm: {args.algo}")
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Mean Reward): {best_trial.value}")

    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    results_df = study.trials_dataframe()
    results_df.to_csv(f"optuna_results_{args.algo}.csv")
    print(f"\nOptuna study results saved to optuna_results_{args.algo}.csv")


if __name__ == '__main__':
    main()