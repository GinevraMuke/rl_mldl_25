import gym
from env.custom_hopper import * 
import numpy as np 

from stable_baselines3 import PPO
from stable_baselines3 import SAC

from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from provaPPO import *


#Creo una classe wrapper per l'environment da utilizzare per randomizzare i parametri nei range desiderati
class UDRWrapper(gym.Wrapper):
    def __init__(self, env, mass_ranges):
        super().__init__(env)
        self.mass_ranges = mass_ranges

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # Randomize thigh, leg, foot masses (indices 2, 3, 4)
        for idx, (low, high) in self.mass_ranges.items():
            self.env.sim.model.body_mass[idx] = np.random.uniform(low, high)

        return obs


#In fase di training posso scegliere se perturbare o meno
def trainingUDR(env, algorithm, tot_timesteps, file_name, use_udr=False):

    base_env = gym.make(env)

    if use_udr:
        # Get default masses from base env
        default_masses = base_env.sim.model.body_mass.copy()
        mass_ranges = {
            2: (0.7 * default_masses[2], 1.3 * default_masses[2]),
            3: (0.7 * default_masses[3], 1.3 * default_masses[3]),
            4: (0.7 * default_masses[4], 1.3 * default_masses[4]),
        }
        randomized_env = UDRWrapper(base_env, mass_ranges)
    else:
        randomized_env = base_env

    training_env = Monitor(randomized_env)

    print('State space:', training_env.observation_space)
    print('Action space:', training_env.action_space)
    print('Dynamics parameters:', training_env.get_parameters())

    model = createModel(algorithm, training_env)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='ppo_hopper'
    )

    model.learn(tot_timesteps, callback=checkpoint_callback)
    model.save(file_name)

    return model



def main():
    
    # TASK 6 — UDR Training
    print("\n--- Training with UDR ---\n")
    udr_model = trainingUDR('CustomHopper-source-v0', 'PPO', 1e5, "ppo_udr", use_udr=True) #Da aggiungere parametro per cambaire distribuzione uniforme

    # Evaluate UDR-trained policy on source and target
    udr_to_source = testing(udr_model, 'CustomHopper-source-v0', 50)
    udr_to_target = testing(udr_model, 'CustomHopper-target-v0', 50)

    printMeanReward("udr->src", udr_to_source[0], udr_to_source[1])
    printMeanReward("udr->tgt", udr_to_target[0], udr_to_target[1])



if __name__ == '__main__':
    main()
