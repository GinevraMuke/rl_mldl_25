import os
from collections import OrderedDict
import torch
import gymnasium as gym
import optuna
from wandb.integration.sb3 import WandbCallback
from wandb.sdk.verify.verify import PROJECT_NAME
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from env_gymnasium.custom_hopper import CustomHopper

#inject function to inject the full garage model
def inject_full_garage_model(ppo_model: PPO, pth_path: str):
    try:
        full_state = torch.load(pth_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il file .pth. Dettagli: {e}")
        return

    if 'policy_state_dict' in full_state:
        actor_state_dict = full_state['policy_state_dict']

        policy_net_dict = OrderedDict()
        policy_net_mapping = {
            '_module._mean_module._layers.0.linear.weight': '0.weight',
            '_module._mean_module._layers.0.linear.bias': '0.bias',
            '_module._mean_module._layers.1.linear.weight': '2.weight',
            '_module._mean_module._layers.1.linear.bias': '2.bias',
        }
        for k_garage, k_sb3 in policy_net_mapping.items():
            if k_garage in actor_state_dict:
                policy_net_dict[k_sb3] = actor_state_dict[k_garage]

        action_net_dict = OrderedDict()
        action_net_mapping = {
            '_module._mean_module._output_layers.0.linear.weight': 'weight',
            '_module._mean_module._output_layers.0.linear.bias': 'bias',
        }
        for k_garage, k_sb3 in action_net_mapping.items():
            if k_garage in actor_state_dict:
                action_net_dict[k_sb3] = actor_state_dict[k_garage]

        try:
            ppo_model.policy.mlp_extractor.policy_net.load_state_dict(policy_net_dict)
            ppo_model.policy.action_net.load_state_dict(action_net_dict)
            print("Policy iniettata con successo.")

        except Exception as e:
            print(f"ERRORE durante l'iniezione dell'attore: {e}")
    else:
        print("[ATTENZIONE] 'policy_state_dict' non trovato nel file .pth.")


    if 'value_function_state_dict' in full_state:
        critic_state_dict = full_state['value_function_state_dict']
        ppo_critic_net = ppo_model.policy.mlp_extractor.value_net

        critic_dict = OrderedDict()
        critic_mapping = {
            '_module._layers.0.linear.weight': '0.weight',
            '_module._layers.0.linear.bias': '0.bias',
            '_module._layers.1.linear.weight': '2.weight',
            '_module._layers.1.linear.bias': '2.bias',

        }
        for k_garage, k_sb3 in critic_mapping.items():
            if k_garage in critic_state_dict:
                critic_dict[k_sb3] = critic_state_dict[k_garage]

        try:
            ppo_critic_net.load_state_dict(critic_dict, strict=False)  # strict=False ignora il layer di output mancante
            print("Value Function iniettata con successo.")
        except Exception as e:
            print(f"ERRORE durante l'iniezione del critico: {e}")
    else:
        print(" 'value_function_state_dict' non trovato. Il critico partirà da zero.")



N_TRIALS = 100            # number of trials for hyperparameter tuning
N_TIMESTEPS_PER_TRIAL = 100000  # timesteps per trial
N_EVAL_EPISODES = 20      # evaluation episodes
EVAL_FREQ = 10000        # frequency of evaluation in timesteps
N_ENVS = 4                # parallel environments for training
ENV_ID = "CustomHopper-target-v0"
PROJECT_NAME = "Hopper-PPO-Tuning-Gymnasium"
INIT_MODEL_PATH = "/Users/ginevramuke/PycharmProjects/rl_mldl_25/MAML_PPO_garage/models/mamlppo_v2.pth"


def objective(trial: optuna.trial.Trial) -> float:

    hyperparams = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [ 512, 1024, 2048]),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 1.0),
        "n_epochs": trial.suggest_int("n_epochs", 5, 20),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.1, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.3),
    }

    total_buffer_size = hyperparams["n_steps"] * N_ENVS
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    while total_buffer_size % batch_size != 0:
        batch_size //= 2
    trial.set_user_attr("batch_size", batch_size)


    run = wandb.init(
        project=PROJECT_NAME,
        config=trial.params,
        group="PPO-Hyperparam-Search",
        name=f"trial-{trial.number}",
        sync_tensorboard=True,
        reinit=True
    )

    vec_env = make_vec_env(ENV_ID, n_envs=N_ENVS, wrapper_class=Monitor)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{run.id}",
        **hyperparams
    )
    inject_full_garage_model(model, INIT_MODEL_PATH)
    eval_env = make_vec_env(ENV_ID, n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        verbose=0
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/wandb/{run.id}",
        verbose=2
    )

    try:
        model.learn(total_timesteps=N_TIMESTEPS_PER_TRIAL, callback=[eval_callback, wandb_callback])
    except (AssertionError, ValueError) as e:
        print(f" {e}")
        run.finish()
        return -1e9

    best_mean_reward = eval_callback.best_mean_reward

    wandb.log({"eval/best_mean_reward": best_mean_reward})
    run.finish()

    vec_env.close()
    eval_env.close()

    return best_mean_reward


def main():
    study = optuna.create_study(direction='maximize')  # we want to maximize reward
    study.optimize(objective, n_trials=100)  #number of research attempt

    print('Best hyperparameters:', study.best_params)


if __name__ == '__main__':
    main()