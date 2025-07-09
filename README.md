# RL for Hopper with Domain Randomization and Meta-Learning

This repository is dedicated to exploring various Reinforcement Learning (RL) techniques for the `CustomHopper` Gym environment. The project is divided into three main directories: `env`, `tasks`, and `project_extension`.

-   `env`: Contains the definition of the custom Hopper environment.
-   `tasks`: Includes implementations of standard RL algorithms like REINFORCE, Actor-Critic, PPO and SAC.
-   `project_extension`: Focuses on our primary research, implementing MAML and other comparison algorithms.

---

## Project Structure

### `env` Directory

This directory contains the necessary files to define and interact with the custom Hopper environment using the MuJoCo physics engine.

-   `custom_hopper.py`: This is the core file of the directory. It defines the custom environment class, its attributes, and methods that are used throughout the other scripts in this project.

### `tasks` Directory

This directory contains implementations of various standard RL algorithms.

#### `task_2_3`

This folder contains scripts for implementing basic algorithms like REINFORCE and Actor-Critic from scratch.

-   `agent_actorCritic.py`: Defines the `Agent` and `Policy` classes for the Actor-Critic algorithm.
-   `agent_Reinforce.py`: Defines the `Agent` and `Policy` classes for the REINFORCE algorithm.
-   `train.py`: A unified script to train either an Actor-Critic or a REINFORCE agent. The algorithm can be selected via a command-line argument.
-   `test.py`: Allows for testing the performance of trained Actor-Critic or REINFORCE models (saved in `.mdl` format).
-   `test_random_policy.py`: A utility script to test the environment configuration with a random policy.

#### `task_4_5`

This folder contains scripts for training and tuning standard algorithms like PPO and SAC, using the `stable-baselines3` library.

-   `train_sb3.py`: Trains a PPO or SAC agent. The algorithm is selectable via a command-line argument.
-   `test_sb3.py`: Tests the models trained with `train_sb3.py`.
-   `tuning_Optuna.py`: A script to perform hyperparameter tuning for PPO or SAC using Optuna.

#### `task_6`

This folder contains scripts for applying Domain Randomization with PPO or SAC.

-   `train_6.py`: Trains an agent within a specified domain randomization interval.
-   `tuning_6.py`: For an agent with fixed hyperparameters, this script finds the optimal domain randomization interval.
-   *Note: Trained models can be tested using `test_sb3.py` from the `task_4_5` directory.*

### `project_extension` Directory

This is the core of our research project
.

#### `ADR_UDR`

This folder contains all the necessary files to train PPO agents with Uniform Domain Randomization (UDR) or Automatic Domain Randomization (ADR).

-   `adr.py`: Contains the class that implements the Automatic Domain Randomization logic.
-   `train.py`: Trains a PPO agent in the *source* environment using either ADR or UDR, selectable via a command-line argument.
-   `finetune.py`: Takes a model pre-trained with ADR or UDR in the *source* environment and performs a few fine-tuning steps in the *target* environment.

#### `garage_MAML`

This folder contains implementations of Model-Agnostic Meta-Learning (MAML) using the `garage` library.

-   `env_gymnasium`: A `gymnasium`-based environment (unlike the `gym`-based ones used elsewhere), which is required by `garage`.
-   `MAMLPPO.py`: Trains a MAML agent using the `garage` library.
-   `tuning_MAML.py`: Performs hyperparameter tuning for the MAML algorithm.
-   `transfer_learning.py`: Takes a model pre-trained with MAML in the *source* environment and fine-tunes a PPO agent for a few steps in the *target* environment.
-   `tuning_ppo.py`: Tunes the hyperparameters of the PPO agent used for fine-tuning in the *target* environment, starting from a pre-trained MAML model.
-   `test_ppo.py`: Visualizes the performance of the final model after transfer learning.

#### `higher_MAML`

This folder contains our custom implementation of MAML, adapted from the `stable-baselines3` source code.

-   `sb3_adapted_classes`: Contains classes adapted from the `stable-baselines3` source code for our custom MAML implementation.
    -   `PPO.py`: The most important script in this sub-directory. It contains the `PPO` class used for MAML training and the `PPO_fine_tune` class used to train a PPO agent in the *target* environment, starting from a pre-trained MAML model.
-   `train_MAML.py`: Trains our custom MAML in the *source* environment.
-   `tuning_MAML.py`: Tunes the hyperparameters for our custom MAML in the *source* environment.
-   `train_MAML_PPO.py`: Trains a PPO agent in the *target* environment, starting from a MAML model pre-trained in the *source* environment.
-   `tuning_MAML_PPO.py`: Tunes the hyperparameters of the PPO agent used for fine-tuning in the *target* environment.
-   `test.py`: Evaluates the performance of the resulting models.