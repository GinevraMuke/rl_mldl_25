import gym
import sys
import os
import numpy as np
import argparse
import random
import wandb

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn
from wandb.integration.sb3 import WandbCallback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *
from adr import AutomaticDomainRandomizer


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default=None, type=str, help='Name of the project on Weights & Biases.')
    parser.add_argument('--env', default="CustomHopper-target-v0", type=str, help='Name of the Gym environment.')
    parser.add_argument('--algo', default="PPO", type=str, help='RL algorithm to use (e.g., PPO, SAC).')
    parser.add_argument('--randomization', default='none', type=str, choices=['none', 'udr', 'adr'], help='Type of domain randomization to use (none, udr, or adr).')
    parser.add_argument('--timesteps', default=400000, type=int, help='Total number of training timesteps.')
    parser.add_argument('--model_destination_path', default='models/best_model', type=str, help='Destination path to save the trained model.')
    parser.add_argument('--save', action='store_true', help='Flag to save the final model.')
    parser.add_argument(
        '--model_config',
        default="{'policy' : 'MlpPolicy', 'verbose' : 1, 'batch_size': 298, 'n_steps': 3859, " \
                "'n_epochs': 27, 'clip_range': 0.14, 'ent_coef': 0.004448319594844347, " \
                "'gae_lambda': 0.928036529859964, 'gamma': 0.9925945837930582, 'use_sde': False, " \
                "'sde_sample_freq': -1, 'vf_coef': 0.5, 'max_grad_norm': 0.5, " \
                "'normalize_advantage': True, 'tensorboard_log': './ppo_hopper_tensorboard/'}",
        type=str,
        help="String dictionary of model hyperparameters. Must follow the default syntax."
    )
    return parser.parse_args()


class ADRCallback(BaseCallback):
    """
    Callback to implement Automatic Domain Randomization (ADR).
    1. In _on_step, it collects rewards from completed episodes.
    2. In _on_rollout_end, it uses the collected rewards to update the ADR manager.
    3. Also in _on_rollout_end, it samples and sets the new environment parameters for the next rollout.
    """
    def __init__(self, verbose: int = 0, perf_threshold_low: float = 200.0, perf_threshold_high: float = 1000.0, 
                 buffer_size: int = 3, step_size: float = 0.2, prob_boundary_test: float = 0.1, max_range_percentage: float = 0.5):
        super().__init__(verbose)
        
        # Configure and initialize the ADR manager
        adr_config = {
            'param_names': ['thigh', 'leg', 'foot'], # Correct order of randomized params
            'initial_values': np.array([3.92699082, 2.71433605, 5.0893801]), # thigh, leg, foot mass
            'perf_threshold_low': perf_threshold_low,
            'perf_threshold_high': perf_threshold_high,
            'buffer_size': buffer_size,
            'step_size': step_size, 
            'max_range_percentage': max_range_percentage 
        }
        self.adr_manager = AutomaticDomainRandomizer(**adr_config)
        
        self.episode_rewards = []
        self.last_boundary_test_details = None
        self.prob_boundary_test = prob_boundary_test

    def _on_step(self) -> bool:
        """Called at every step. Used to capture rewards at the end of an episode."""
        # The Monitor wrapper adds 'episode' to the 'info' dict when an episode ends.
        # 'dones' is an array, one for each parallel env (in our case, only one).
        if self.locals["dones"][0] and "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(ep_reward)
            self.logger.record("adr/episode_reward", ep_reward)
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout to update and apply ADR."""
        # 1. Update ADR with the performance from the last rollout
        if self.last_boundary_test_details and self.episode_rewards:
            avg_perf = np.mean(self.episode_rewards)
            param_name, boundary_type = self.last_boundary_test_details
            self.adr_manager.update(param_name, boundary_type, avg_perf)
        self.episode_rewards.clear()

        # 2. Sample new randomized parameters (thigh, leg, foot)
        if random.random() < self.prob_boundary_test:
            randomized_params, param, boundary = self.adr_manager.boundary_sample()
            self.last_boundary_test_details = (param, boundary)
        else:
            randomized_params = self.adr_manager.sample_env()
            self.last_boundary_test_details = None

        # 3. Rebuild the full parameter vector and set it in the environment
        # Get the current full parameter vector (4 elements)
        full_task_params = self.training_env.envs[0].unwrapped.get_parameters()
        # Overwrite the last 3 elements with the randomized ones, leaving the torso mass intact.
        full_task_params[1:] = randomized_params
        # Set the updated vector in the environment for the next rollout
        self.training_env.envs[0].unwrapped.set_parameters(full_task_params)
        
        # 4. Log ADR and environment data
        self.logger.record("adr/entropy", self.adr_manager.get_entropy())
        current_params = self.training_env.envs[0].unwrapped.get_parameters()
        full_param_names = ['torso', 'thigh', 'leg', 'foot'] 
        
        for i, name in enumerate(full_param_names):
            self.logger.record(f"env_params/{name}", current_params[i])
        
        if self.verbose > 0:
            params_dict_for_printing = {name: current_params[i] for i, name in enumerate(full_param_names)}
            print(f"New env params for next rollout: {params_dict_for_printing}")


class UDRCallback(BaseCallback):
    """A simple callback to apply Uniform Domain Randomization at the end of each rollout."""
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Sets new random parameters in the environment."""
        # This assumes the environment's `set_random_parameters` method implements UDR.
        if hasattr(self.training_env, 'envs'):
            self.training_env.envs[0].set_random_parameters()
            if self.verbose > 0:
                print(f"New random UDR parameters set: {self.training_env.envs[0].get_parameters()}")


def print_final_adr_report(adr_manager: AutomaticDomainRandomizer):
    """Prints a final, well-formatted report of the ADR intervals, including percentage changes."""
    print("\n" + "="*85)
    print("||" + " Final ADR Intervals Report ".center(81) + "||")
    print("="*85)
    
    if not hasattr(adr_manager, 'initial_values'):
        print("ERROR: Could not generate report. `initial_values` not found in ADR manager.")
        return

    initial_values = adr_manager.initial_values
    param_names = adr_manager.param_names
    low_bounds = adr_manager.low_bounds
    high_bounds = adr_manager.high_bounds

    print(f"{'Parameter':<12} | {'Initial Val':<15} | {'Final Range':<28} | {'Change % (min/max)':<25}")
    print("-" * 85)

    for i, name in enumerate(param_names):
        initial = initial_values[i]
        low = low_bounds[i]
        high = high_bounds[i]

        # Calculate percentage change, handling division by zero
        if initial != 0:
            low_perc = ((low - initial) / abs(initial)) * 100
            high_perc = ((high - initial) / abs(initial)) * 100
        else:
            low_perc = float('inf') if low != 0 else 0
            high_perc = float('inf') if high != 0 else 0

        initial_str = f"{initial:.4f}"
        range_str = f"[{low:.4f}, {high:.4f}]"
        perc_str = f"[{low_perc:+.2f}%, {high_perc:+.2f}%]"

        print(f"{name:<12} | {initial_str:<15} | {range_str:<28} | {perc_str:<25}")
    
    print("-" * 85)


def create_model(args: argparse.Namespace, env):
    """Creates a PPO or SAC model based on the provided arguments."""
    if args.algo == 'PPO':
        model = PPO(
            env=env,
            **eval(args.model_config),
            learning_rate=get_linear_fn(0.002, 1e-5, 1.0),
            device='cpu'
        )
    elif args.algo == 'SAC':
        model = SAC(
            env=env,
            **eval(args.model_config)
        )
    else:
        raise ValueError(f"Algorithm '{args.algo}' not supported. Choose 'PPO' or 'SAC'.")
    return model


def train(args: argparse.Namespace):
    """Main training function."""
    # Create the training environment, wrapped in a Monitor
    training_env = Monitor(gym.make(args.env, udr_ranges={'thigh': 0.5, 'leg': 0.5, 'foot': 0.5}))
    
    print('State space:', training_env.observation_space)
    print('Action space:', training_env.action_space)
    print('Initial dynamics parameters:', training_env.get_parameters())

    model = create_model(args, training_env)
    
    # Set up callbacks based on the chosen randomization method
    callback_list = []
    run = None
    adr_manager = None
    
    if args.project_name:
        run = wandb.init(
            project=args.project_name,
            config=eval(args.model_config),
            sync_tensorboard=True
        )
        callback_list.append(WandbCallback())

    if args.randomization == 'adr':
        print("\n--- Training with Automatic Domain Randomization (ADR) ---")
        adr_callback = ADRCallback(verbose=1, max_range_percentage=0.7)
        callback_list.append(adr_callback)
        adr_manager = adr_callback.adr_manager # For the final report
    elif args.randomization == 'udr':
        print("\n--- Training with Uniform Domain Randomization (UDR) ---")
        udr_callback = UDRCallback(verbose=1)
        callback_list.append(udr_callback)
    else:
        print("\n--- Training with standard PPO (no randomization) ---")

    # Train the model
    model.learn(args.timesteps, callback=CallbackList(callback_list) if callback_list else None)

    # Print the final ADR report only if ADR was used
    if args.randomization == 'adr' and adr_manager:
        print_final_adr_report(adr_manager)
    
    if run:
        run.finish()

    # Save the final model if requested
    if args.save:
        model.save(args.model_destination_path)
        print(f"Model saved to {args.model_destination_path}")

    return model


if __name__ == '__main__':
    args = parse_args()
    train(args)