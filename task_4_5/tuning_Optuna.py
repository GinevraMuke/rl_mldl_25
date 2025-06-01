import optuna
import wandb
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='PPO', type=str, help='Model name')
    parser.add_argument('--project_name', default= None, type=str, help='Name of the project on wandb')
    parser.add_argument('--timesteps', default= 500000, type=int, help='timesteps of PPO form sb3')


    return parser.parse_args()

args = parse_args()

def optimize(trial):

    learning_rate = trial.suggest_float('learning_rate', 5e-6, 0.003)
    batch_size = trial.suggest_int('batch_size', 4, 4096)
    n_steps = trial.suggest_int('n_steps', 32, 5000)
    n_epochs = trial.suggest_int('n_epochs', 3, 30)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_float('ent_coef', 0, 0.01)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 1)
    gamma = trial.suggest_float('gamma', 0.8, 0.99997)

    run = wandb.init(
        project= args.project_name,
        id=str(trial.number),
        sync_tensorboard=True,
        config= dict(trial.params),
        reinit=True
    )

    env = Monitor(gym.make('CustomHopper-source-v0'))

    model = PPO('MlpPolicy', env, learning_rate=learning_rate, n_steps=n_steps,
                batch_size=batch_size, n_epochs=n_epochs, clip_range=clip_range, ent_coef= ent_coef, gae_lambda= gae_lambda, gamma = gamma,tensorboard_log="./ppo_tensorboard_logs/")

    model.learn(total_timesteps= int(1e5), callback=WandbCallback())

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    wandb.log({"mean_reward_eval": mean_reward})

    run.finish()

    return mean_reward






def main():
    study = optuna.create_study(direction='maximize')  # we want to maximize reward
    study.optimize(optimize, n_trials=100)  #number of research attempt

    # Stampa i migliori parametri
    print('Best hyperparameters:', study.best_params)



if __name__ == '__main__':
    main()