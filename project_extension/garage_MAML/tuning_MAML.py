import sys
import os
import warnings
import math


warnings.filterwarnings('ignore', module='garage._dtypes')
warnings.filterwarnings('ignore', module='gymnasium.spaces.box')

import gymnasium as gym
sys.modules['gym'] = gym

import torch
import optuna
import wandb

from garage.experiment.deterministic import set_seed
from garage.experiment.snapshotter import SnapshotConfig
from garage.envs import GymEnv
from garage.sampler import LocalSampler
from garage.experiment.task_sampler import ConstructEnvsSampler
from garage.trainer import Trainer
from garage.torch.algos import MAMLPPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


from env_gymnasium.custom_hopper import CustomHopper
max_episode_steps = 500

#wrapper from gymnasium to gym
class GymnasiumToGymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        md = getattr(env, "metadata", {})
        self.metadata = {"render.modes": md.get("render_modes", ["human", "rgb_array"])}

    def reset(self, **kwargs):
        obs, _info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        return obs, reward, done, info


def make_env(max_episode_steps: int = max_episode_steps) :
    raw = CustomHopper(domain="source",
                       xml_file="hopper.xml",
                       frameskip=4)
    # Abilita UDR e campiona parametri ad ogni reset
    raw.udr_enabled = True
    # Fai un primo campionamento diretto
    raw.set_random_parameters()
    # Patch metadata per GymEnv
    raw.metadata["render.modes"] = raw.metadata.get("render_modes",
                                                    ["human", "rgb_array"])
    wrapped = GymnasiumToGymWrapper(raw)
    return GymEnv(wrapped, max_episode_length=max_episode_steps)

#objective for optuna, with the values that we wanted to try
def objective(trial):
    # 1) Sampling iperparametri (originari)
    batch_size      = trial.suggest_categorical("batch_size",    [256, 512, 1024])
    meta_batch_size = trial.suggest_int(       "meta_batch_size", 10, 50, step=10)
    num_inner       = trial.suggest_int(       "num_inner_grad_steps", 1, 5)
    inner_lr        = trial.suggest_loguniform("inner_lr",        1e-2,   1.0)
    outer_lr        = trial.suggest_loguniform("outer_lr",        1e-4,   1e-2)
    clip_range      = trial.suggest_uniform(   "clip_range",      0.1,    0.3)
    entropy_coeff   = trial.suggest_uniform(   "entropy_coeff",   0.0,    0.05)
    discount        = trial.suggest_uniform(   "discount",        0.9,    0.999)
    gae_lambda      = trial.suggest_uniform(   "gae_lambda",      0.8,    0.99)
    hidden_exp      = trial.suggest_int(       "hidden_size_exp", 6,      8)
    hidden_sizes    = (2**hidden_exp, 2**hidden_exp)


    run = wandb.init(
        project="Optuna-tuning-MAMLPPO",
        name=f"trial_{trial.number}",
        job_type="optuna",
        config={
            "batch_size": batch_size,
            "meta_batch_size": meta_batch_size,
            "num_inner_grad_steps": num_inner,
            "inner_lr": inner_lr,
            "outer_lr": outer_lr,
            "clip_range": clip_range,
            "entropy_coeff": entropy_coeff,
            "discount": discount,
            "gae_lambda": gae_lambda,
            "hidden_sizes": hidden_sizes,
        },
        reinit=True,
    )


    seed = 42
    set_seed(seed)
    max_steps = 500
    single_env = make_env(max_steps)
    env_spec   = single_env.spec

    #policy and value function
    policy = GaussianMLPPolicy(
        env_spec,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=torch.tanh,
        learn_std=True,
        std_parameterization='exp',
        init_std=1.0
    )
    MIN_STD_VALUE = 0.1
    policy._min_std = torch.tensor(MIN_STD_VALUE)

    vf = GaussianMLPValueFunction(
        env_spec,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=torch.tanh,
    )


    sampler_envs = [make_env(max_steps) for _ in range(4)]
    sampler      = LocalSampler(
        agents=policy,
        envs=sampler_envs,
        max_episode_length=max_steps,
        n_workers=4,
        seed=seed,
    )


    task_sampler = ConstructEnvsSampler([make_env] * meta_batch_size)


    algo = MAMLPPO(
        single_env,
        policy,
        vf,
        sampler,
        task_sampler,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        lr_clip_range=clip_range,
        discount=discount,
        gae_lambda=gae_lambda,
        policy_ent_coeff=entropy_coeff,
        entropy_method='regularized',
        meta_batch_size=meta_batch_size,
        num_grad_updates=num_inner,
    )


    snapshot_dir = os.path.join(os.getcwd(), "optuna_snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_config = SnapshotConfig(snapshot_dir=snapshot_dir,
                                     snapshot_mode='last',
                                     snapshot_gap=1)
    trainer = Trainer(snapshot_config=snapshot_config)
    trainer.setup(algo=algo, env=single_env)

    #train for 100 epochs
    avg_return = 0.0
    try:
        for epoch in range(1, 100):
            avg_return = trainer.train(n_epochs=1, batch_size=batch_size)
            if not math.isfinite(avg_return):
                raise RuntimeError("avg_return is NaN/Inf → pruning")
            wandb.log({"epoch": epoch, "avg_return": float(avg_return)})
            trial.report(avg_return, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    except Exception:
        run.finish()
        raise optuna.TrialPruned()

    run.finish()
    return float(avg_return)

#save the best model
if __name__ == "__main__":

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=40),
    )
    study.optimize(objective, n_trials=20)


    print("→ Best trial:", study.best_trial.number)
    print("→ Best value:", study.best_value)
    print("→ Best params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    wandb.finish()


