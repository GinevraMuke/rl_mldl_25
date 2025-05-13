"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3 import SAC

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback


# Create environment, train and save model
def training(env, algorithm, tot_timesteps, file_name, progect_name : str = ""):
    if progect_name != "":
        #configuration of wandb
        run = wandb.init(
            project= progect_name,
            id = "trial run",
            sync_tensorboard = True
        )



    training_env = Monitor(gym.make(env))

    print('State space:', training_env.observation_space)
    print('Action space:', training_env.action_space)
    print('Dynamics parameters:', training_env.get_parameters())

    model = createModel(algorithm, training_env)

    # Optional: Save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='ppo_hopper'
    )

    # Training and saving results
    if progect_name != "": #add callback for wandb
        model.learn(tot_timesteps, callback= WandbCallback())
        run.finish()
    else:
        model.learn(tot_timesteps, callback=checkpoint_callback)

    model.save(file_name)

    return model


def testing(model, env, n_episodes):
    testing_env = gym.make(env)
    mean_reward, std_reward = evaluate_policy(model, testing_env, n_eval_episodes=n_episodes)
    return mean_reward, std_reward


def printMeanReward(label, mean, std):
    print(f"{label:<30} → Avg. Return: {mean:.2f} ± {std:.2f}")


def createModel(algorithm, env):
    if algorithm == 'PPO':

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            tensorboard_log="./ppo_hopper_tensorboard/"
        )

    elif algorithm == 'SAC':
        # Aggiungi parametri
        model = SAC("MlpPolicy", env, verbose=1)

    else:
        raise ValueError("Choose 'PPO' or 'SAC'")

    return model


def main():
    """
    #TASK 4
    model = training('CustomHopper-source-v0', 'PPO', 1e5, "training_results")
    mean, devstd = testing(model, 'CustomHopper-source-v0', 50)
    printMeanReward("firstTest", mean, devstd)

    """
    # Ho commentato task 4 perchè tanto rifacciamo gli stessi calcoli dopo

    # TASK 5
    # Creating models
    source_model = training('CustomHopper-source-v0', 'PPO', 1e5, "/home/ginevramuke/rl_mldl_25/models/ppo_source", "ppo_source")
    target_model = training('CustomHopper-target-v0', 'PPO', 1e5, "/home/ginevramuke/rl_mldl_25/models/ppo_target", "ppo_target")

    # Evaluating every combination of models
    source_to_source = testing(source_model, 'CustomHopper-source-v0', 50)
    source_to_target = testing(source_model, 'CustomHopper-target-v0', 50)
    target_to_target = testing(target_model, 'CustomHopper-target-v0', 50)

    # Printing
    printMeanReward("src->src", source_to_source[0], source_to_source[1])
    printMeanReward("src->tgt", source_to_target[0], source_to_target[1])
    printMeanReward("tgt->tgt", target_to_target[0], target_to_target[1])


if __name__ == '__main__':
    main()