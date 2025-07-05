
import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from env_gymnasium.custom_hopper import CustomHopper


def parse_args():
    parser = argparse.ArgumentParser(description="Testa un modello PPO addestrato su CustomHopper.")

    parser.add_argument(
        "--model-path",
        type=str,
        default="/Users/ginevramuke/PycharmProjects/rl_mldl_25/MAML_PPO_garage/models/ppo_bello/final_model.zip",
        required=False,
        help="path of the model (es. 'models/.../best_model.zip')."
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="CustomHopper-target-v0", # Cambia questo se vuoi testare un altro ambiente di default
        help="ID of the enviroment you choose."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="episodes to watch during render."
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic"

    )
    parser.set_defaults(deterministic=True)

    return parser.parse_args()


def main():

    args = parse_args()

    try:
        env = gym.make(args.env_id, render_mode="human")
    except Exception as e:
        print(f"{e}")
        return


    try:
        model = PPO.load(args.model_path, env=env)
    except Exception as e:
        print(f"{e}")
        env.close()
        return

    total_rewards = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            env.render()


        print(f"Episode {episode + 1:2d} | Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    env.close()

    if total_rewards:
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        print("--- Results---")
        print(f"AVG Reward on {args.episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}")


if __name__ == '__main__':
    main()