import optuna
import argparse
import wandb
from train_MAML import train


def objective(trial, args):
    try:
        run = wandb.init(
            project=args.project_name,
            config=trial.params,
            name=f"trial-{trial.number}",
        )

        params = {
            # Parameters MAML
            'inner_lr': trial.suggest_float("inner_lr", 1e-4, 5e-2, log=True),
            'outer_lr': trial.suggest_float("outer_lr", 1e-5, 1e-3, log=True),
            'inner_updates': trial.suggest_int("inner_updates", 1, 10),

            # Parameters PPO
            'num_steps': trial.suggest_categorical("num_steps", [256, 512, 1024]),
            'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            'num_epochs': trial.suggest_int("num_epochs", 3, 20),
            'clip_range': trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4]),
            'vf_coef': trial.suggest_float("vf_coef", 0.3, 1.0),
            'ent_coef': trial.suggest_float("ent_coef", 1e-5, 1e-2, log=True),
            'gae_lambda': trial.suggest_float("gae_lambda", 0.9, 0.999),
            'gamma': trial.suggest_float("gamma", 0.95, 0.9999),

            # Parameters of env and net
            'num_task': trial.suggest_int("num_task", 10, 30),
            'hidden_size': trial.suggest_categorical("hidden_size", [64, 128, 256]),
            'dr_coef': trial.suggest_float("dr_coef", 0.1, 0.5),

            # the unique fixed hyperparameter fot the tuning
            'num_meta_iterations': 100,
        }

        # update wandb with suggested parameters
        wandb.config.update(params)

        # argparse for train
        train_args = argparse.Namespace(
            isTerminal=False, project_name=None, model_destination_path='',
            save=False, model_config=str(params)
        )

        print(f"\n--- start Trial #{trial.number} (Run WandB: {run.name}) ---")

        # start trial
        final_reward = train(train_args, trial=trial)

        return final_reward


    except optuna.TrialPruned:

        raise


    finally:

        if wandb.run is not None:
            wandb.finish()


def main(args):
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)

    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=sampler)

    start_baseline = {
        'inner_lr': 3e-3,
        'outer_lr': 3e-4,
        'inner_updates': 2,
        'num_epochs': 4,
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'ent_coef': 0.01,
        'gae_lambda': 0.99,
        'gamma': 0.99,
        'num_task': 10,
        'hidden_size': 64,
        'dr_coef': 0.1,
        'batch_size': 64,
        'num_steps': 1024,
    }
    #start the study with a notorious configuration
    study.enqueue_trial(start_baseline)

    objective_with_args = lambda trial: objective(trial, args)

    try:
        study.optimize(objective_with_args, n_trials=50, n_jobs=1)
    except KeyboardInterrupt:
        print("Tuning interrupted.")

    print("\n\n--- TUNING COMPLETED ---")

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (Max Mean Reward): {best_trial.value:.4f}")

    print("  Best hyperparams founded:")
    best_params = best_trial.params.copy()
    print(best_params)

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default="MAML-PPO-Full-Tuning", type=str, help='Name of the project on wandb')
    return parser.parse_args()


if __name__ == '__main__':
    cli_args = parse_cli_args()
    main(cli_args)