import optuna
import argparse
import wandb
import torch
import random
import numpy as np
from optuna.integration.wandb import WeightsAndBiasesCallback
from train_MAML_PPO import train as train_ppo_finetune


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default= "PPO_MAML tuning 50k bumbling_forest", type=str, help='Name of the project on wandb')
    parser.add_argument('--model', default= "models_extension/bumbling_forest.mdl", type=str, help='MAML model to be fine tuned')
    return parser.parse_args()

def objective(trial: optuna.Trial):
    args = parse_args()
    """
    Questa funzione viene eseguita da Optuna per ogni trial.
    Definisce gli iperparametri da provare, esegue il training e ritorna una metrica.
    """
    print(f"\n--- Trial #{trial.number} ---")

    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'num_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'num_epochs': trial.suggest_int('n_epochs', 5, 20),
        'gamma': 0.99, #generally fixed
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'vf_coef': 0.5, #generally fixed
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-3, log=True),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
        'hidden_size': 256,
    }

    run = wandb.init(
        project=args.project_name,
        id=f"run_{trial.number}",
        config=hyperparams,
        reinit = True,
        sync_tensorboard = True
    )

    #create the object NameSpace to pass to the train function
    argp = argparse.Namespace(
        isTerminal=False,
        project_name=args.project_name,
        env="CustomHopper-target-v0",
        model=args.model,
        model_destination_path=f'models_extension/trial2_{trial.number}_model.mdl',
        save=True,
        model_config=str(hyperparams)
    )

    mean_reward = train_ppo_finetune(argp, wandb_run= run)

    run.finish()

    return mean_reward

#execute the study of optuna
if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    # print final results
    print("\n--- End Tuning ---")
    print("Miglior trial:")
    trial = study.best_trial
    print(f"  with value (Mean Reward): {trial.value:.4f}")
    print("  Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
