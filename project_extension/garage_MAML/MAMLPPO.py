import sys
import os
import warnings

warnings.filterwarnings('ignore', module='garage._dtypes')
warnings.filterwarnings('ignore', module='gymnasium.spaces.box')

import gymnasium as gym
sys.modules['gym'] = gym
import torch
import wandb
import numpy as np
from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.sampler import LocalSampler
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from env_gymnasium.custom_hopper import CustomHopper

#Wrapper to convert gymnasium environments to gym environments
class GymnasiumToGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        md = getattr(env, "metadata", {})
        self.metadata = {"render.modes": md.get("render_modes", ["human", "rgb_array"]) }

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


@wrap_experiment(snapshot_mode='all', archive_launch_repo=False)
def maml_ppo_custom_hopper(ctxt,
                           seed: int = 42,
                           n_epochs: int = 300,
                           batch_size: int = 1024,
                           max_episode_steps: int = 500,
                           meta_batch_size: int = 15,
                           num_inner_grad_steps: int = 4,
                           inner_lr: float = 0.0015315754665763555,
                           outer_lr: float = 0.0001492196637379901,
                           discount: float = 0.9948861106177456,
                           gae_lambda: float = 0.9882076019901432,
                           clip_range: float = 0.1163380577258028,
                           entropy_coeff: float = 0.01,
                           hidden_sizes=(256, 256)):

    config = locals().copy()
    del config['ctxt']
    set_seed(seed)
    wandb.init(
        project="MAMLPPO-policy",
        config=config,
        reinit=True,
    )


    def make_env():
        raw = CustomHopper(domain="source",
                           xml_file="hopper.xml",
                           frameskip=4)
        raw.udr_enabled = True
        raw.metadata["render.modes"] = raw.metadata.get("render_modes",
                                                         ["human", "rgb_array"])

        wrapped = GymnasiumToGymWrapper(raw)
        garage_env = GymEnv(wrapped, max_episode_length=max_episode_steps)

        return garage_env

    single_env = make_env()
    env_spec = single_env.spec

    #Policy and Value Function from Garage
    policy = GaussianMLPPolicy(
        env_spec,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=torch.tanh,
        learn_std=True,
        std_parameterization='exp',
        init_std=1.0
    )
    #std for better exploration
    MIN_STD_VALUE = 0.1
    policy._min_std = torch.tensor(MIN_STD_VALUE)

    value_function = GaussianMLPValueFunction(
        env_spec,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=torch.tanh,
    )


    # Rollout sampler, with parallel workers
    n_workers = 4
    sampler_envs = [make_env() for _ in range(n_workers)]
    rollout_sampler = LocalSampler(
        agents=policy,
        envs=sampler_envs,
        max_episode_length=max_episode_steps,
        n_workers=n_workers,
        seed=seed,
    )

    # Task sampler, which creates multiple environments for meta-learning
    task_sampler = ConstructEnvsSampler([make_env] * meta_batch_size)

    #MAML-PPO
    algo = MAMLPPO(
        single_env,
        policy,
        value_function,
        rollout_sampler,
        task_sampler,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        lr_clip_range=clip_range,
        discount=discount,
        gae_lambda=gae_lambda,
        policy_ent_coeff=entropy_coeff,
        entropy_method='regularized',
        meta_batch_size=meta_batch_size,
        num_grad_updates=num_inner_grad_steps,
    )

    #TRAINING LOOP
    trainer = Trainer(snapshot_config=ctxt)
    trainer.setup(algo=algo, env=single_env)

    for epoch in range(1, n_epochs + 1):
        avg_return = trainer.train(n_epochs=1, batch_size=batch_size)
        wandb.log({
            'epoch': epoch,
            'average_return': float(avg_return),
        })

    #save model, with both policiy and value function
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    full_state = {
        'policy_state_dict': policy.state_dict(),
        'value_function_state_dict': value_function.state_dict()
    }

    model_filename = f"mamlppo_v2.pth"
    model_path = os.path.join(model_dir, model_filename)

    torch.save(full_state, model_path)
    wandb.save(model_path)

if __name__ == "__main__":
    maml_ppo_custom_hopper(seed=42)
