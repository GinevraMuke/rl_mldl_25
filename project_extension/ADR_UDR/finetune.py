import optuna
import wandb
import sys
import os
import torch
import gym
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from env.custom_hopper import *
from stable_baselines3 import PPO, SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_linear_fn

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='PPO', type=str, help='RL algorithm to use.')
    parser.add_argument('--env', default="CustomHopper-target-v0", type=str, help='Name of the Gym environment.')
    parser.add_argument('--project_name', default=None, type=str, help='Name of the project on Weights & Biases.')
    parser.add_argument('--timesteps', default=100000, type=int, help='Timesteps for fine-tuning.')
    parser.add_argument('--pre_trained_model', default="models/best_maml.mdl", type=str, help='Path to the pre-trained model (.zip or .mdl).')
    parser.add_argument('--save', action='store_true', help='Flag to save the final model.')
    parser.add_argument('--model_destination_path', default='models/fine_tuned_model', type=str, help='Destination path for the saved model.')
    parser.add_argument(
        '--model_config',
        default="{'policy': 'MlpPolicy', 'verbose': 1, 'batch_size': 298, 'n_steps': 3859, " \
                "'n_epochs': 27, 'clip_range': 0.14, 'ent_coef': 0.004448319594844347, " \
                "'gae_lambda': 0.928036529859964, 'gamma': 0.9925945837930582, 'use_sde': False, " \
                "'sde_sample_freq': -1, 'vf_coef': 0.5, 'max_grad_norm': 0.5, " \
                "'normalize_advantage': True, 'tensorboard_log': './ppo_hopper_tensorboard/'}",
        type=str,
        help="String dictionary of model hyperparameters."
    )
    return parser.parse_args()

def create_model(args: argparse.Namespace, env):
    """Creates a PPO or SAC model based on the provided arguments."""
    if args.algo == 'PPO':
        model = PPO(
            env=env,
            **eval(args.model_config),
            learning_rate=get_linear_fn(1e-4, 1e-5, 1.0), # Learning rate for fine-tuning
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
    """Main training and fine-tuning function."""
    
    # Create the training environment, wrapped in a Monitor
    training_env = Monitor(gym.make(args.env))

    print('State space:', training_env.observation_space)
    print('Action space:', training_env.action_space)
    print('Dynamics parameters:', training_env.get_parameters())

    # --- Automatic Model Loading Logic ---
    model = None
    if args.pre_trained_model.endswith('.zip'):
        # Case 1: Load a full Stable-Baselines3 model from a .zip file
        print(f"Loading pre-trained SB3 model from: {args.pre_trained_model}")
        model = PPO.load(args.pre_trained_model, env=training_env)
        # Reset the learning rate for the fine-tuning phase
        model.learning_rate = get_linear_fn(1e-4, 1e-5, 1.0)

    elif args.pre_trained_model.endswith('.mdl'):
        # Case 2: Transfer weights from a PyTorch state_dict (.mdl file), e.g., from SAC to PPO
        print(f"Transferring weights from state_dict: {args.pre_trained_model}")
        
        # Create the destination PPO model with a random initialization
        if args.algo != 'PPO':
            raise ValueError("Weight transfer is configured for a PPO target model. Please run with --algo PPO.")
        ppo_model = create_model(args, training_env)
        
        # Load the source weights from the .mdl file
        source_weights = torch.load(args.pre_trained_model, map_location=ppo_model.device)
        
        # Define the mapping from SAC layer names to PPO layer names
        # This is necessary because the actor and critic architectures have different names.
        key_mapping = {
            # Actor mapping (SAC -> PPO)
            "actor_net.0.weight": "mlp_extractor.policy_net.0.weight", "actor_net.0.bias": "mlp_extractor.policy_net.0.bias",
            "actor_net.2.weight": "mlp_extractor.policy_net.2.weight", "actor_net.2.bias": "mlp_extractor.policy_net.2.bias",
            "actor_mean.weight": "action_net.weight", "actor_mean.bias": "action_net.bias",
            
            # Critic mapping (SAC -> PPO)
            # We use one of SAC's two critics to populate PPO's single value network.
            "critic_net.0.weight": "mlp_extractor.value_net.0.weight", "critic_net.0.bias": "mlp_extractor.value_net.0.bias",
            "critic_net.2.weight": "mlp_extractor.value_net.2.weight", "critic_net.2.bias": "mlp_extractor.value_net.2.bias",
            "critic_value.weight": "value_net.weight", "critic_value.bias": "value_net.bias",
        }
        
        # Create a new state_dict for the PPO model and populate it with the mapped weights
        ppo_state_dict = ppo_model.policy.state_dict()
        for src_key, dest_key in key_mapping.items():
            if src_key in source_weights and dest_key in ppo_state_dict:
                ppo_state_dict[dest_key] = source_weights[src_key]
        
        # Load the new "hybrid" state_dict into the PPO model
        ppo_model.policy.load_state_dict(ppo_state_dict)
        print("Weight transfer complete.")
        model = ppo_model
    else:
        raise ValueError(f"Unsupported model format: {args.pre_trained_model}. Please use a .zip or .mdl file.")

    # --- End of Loading Logic ---

    # Set up Weights & Biases if a project name is provided
    run = None
    if args.project_name:
        run = wandb.init(
            project=args.project_name,
            config=eval(args.model_config),
            sync_tensorboard=True
        )
    
    # Select callback for training
    callback = WandbCallback() if args.project_name else None
    
    # Fine-tune the model
    print(f"\nStarting fine-tuning for {args.timesteps} timesteps...")
    model.learn(args.timesteps, callback=callback)
    print("Fine-tuning finished.")

    if run:
        run.finish()

    # Save the final fine-tuned model if requested
    if args.save:
        model.save(args.model_destination_path)
        print(f"Model saved to {args.model_destination_path}")

    return model

def main():
    """Main execution function."""
    args = parse_args()
    train(args)

if __name__ == '__main__':
    main()