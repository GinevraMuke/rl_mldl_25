
import os
from collections import OrderedDict
import argparse
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn

from env_gymnasium.custom_hopper import CustomHopper

#function to inject the full garage model (actor + critic) into a PPO model
def inject_full_garage_model(ppo_model: PPO, pth_path: str):
    try:
        full_state = torch.load(pth_path, map_location='cpu', weights_only=True)
    except Exception as e:
        print(f"{e}")
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
            print("Policy injected.")

        except Exception as e:
            print(f"{e}")



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
            print("Value Function injected.")
        except Exception as e:
            print(f"{e}")



def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning of PPO.")
    parser.add_argument("--env-id", type=str, default="CustomHopper-target-v0", help="enviroment ID.")
    parser.add_argument("--timesteps", type=int, default=100000, help="Timestep for fine-tuning.")
    parser.add_argument("--n-envs", type=int, default=4, help="number of parallel env.")
    parser.add_argument(
        "--init-model-path",
        type=str,
        default="/Users/ginevramuke/PycharmProjects/rl_mldl_25/MAML_PPO_garage/models/mamlppo_v2.pth",
        required=False
    )
    return parser.parse_args()


def main():
    args = parse_args()
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    vec_env = make_vec_env(args.env_id, n_envs=args.n_envs, seed=0)
    # model with parameters found in the tuning
    model = PPO(
        learning_rate= 0.00009112325862453216,
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="tensorboard_logs/",
        n_steps=2048,
        batch_size=32,
        n_epochs= 14,
        gamma=0.9944303132334704,
        gae_lambda=0.9998237003742282,
        clip_range=0.24233002169507223,
        ent_coef= 0.00004050457048035798,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None
    )

    inject_full_garage_model(model, args.init_model_path)
    run_name_suffix = "_from_FULL_GARAGE"
    save_dir = f"models/ppo_bello"
    os.makedirs(save_dir, exist_ok=True)

    eval_env = gym.make(args.env_id)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, 'best_model'),
        log_path=os.path.join(save_dir, 'logs'),
        eval_freq=max(10000 // args.n_envs, 1),
        deterministic=True
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=f"{args.env_id}{run_name_suffix}"
    )

    final_model_path = os.path.join(save_dir, 'final_model.zip')
    model.save(final_model_path)

    vec_env.close()
    eval_env.close()

if __name__ == '__main__':
    main()