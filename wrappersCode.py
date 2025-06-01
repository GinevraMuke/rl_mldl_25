import gym
import mujoco_py
from gym.envs.mujoco import HopperEnv

import numpy as np
from gym import Wrapper

#Le varie funzioni setDiff permettono di cambaire inclinazione dinamicamente durante training


#in realtà modifica la direzione del vettore gravitazionale
#per simulare il cammino su piano inclinato
class SlopeWrapper(Wrapper): 

    def __init__(self, env, slope_angle=0.05):
        super().__init__(env)
        self.slope_angle = slope_angle
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._apply_slope()
        return obs
    
    def _apply_slope(self):
        #Quindi bisogna calcolare la componente sui due assi
        angle = self.slope_angle
        self.env.model.opt.gravity[0] = -9.81 * np.sin(angle)  # x-axis
        self.env.model.opt.gravity[2] = -9.81 * np.cos(angle)  # z-axis

    def set_difficulty(self, difficulty):
        if 'slope' in difficulty:
            self.slope_angle = difficulty['slope']
            self._apply_slope()



#Vento creato applicando una forza orizzontale costante, variabile durante allenamento
class WindWrapper(Wrapper):
    def __init__(self, env, force_magnitude=10):
        super().__init__(env)
        self.force_magnitude = force_magnitude

    def set_difficulty(self, difficulty):
        if 'wind' in difficulty:
            self.force_magnitude = difficulty['wind']

    def step(self, action):
        # Apply a horizontal force at the torso
        self.env.sim.data.xfrc_applied[self.env.model.body_name2id('torso'), 0] = self.force_magnitude
        return self.env.step(action)



#Aggiungendo massa alla coscia per aumentare la difficoltà
#La extra mass si resetta ad ogni episodio

#Non ricordavo sinceramente se era una roba che potevamo modificare durante
#Le richieste vecchie, ma per modularità e facilità nel creare 
#Ambeinte wrapper ho fatto comuqnue una classe
class LegWeightWrapper(Wrapper):
    def __init__(self, env, extra_mass=5.0):
        super().__init__(env)
        self.extra_mass = extra_mass
        self.original_mass = None
        self.geom_name = 'thigh_geom'  # You can make this configurable

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._apply_extra_mass()
        return obs

    def _apply_extra_mass(self):
        geom_id = self.env.model.geom_name2id(self.geom_name)
        if self.original_mass is None:
            self.original_mass = self.env.model.body_mass[geom_id]
        self.env.model.body_mass[geom_id] = self.original_mass + self.extra_mass

    def set_difficulty(self, difficulty):
        if 'extra_mass' in difficulty:
            self.extra_mass = difficulty['extra_mass']
            self._apply_extra_mass()



#Per fare curriculum, cambiando parametri ambiente durante gli epsiodi
#schedule_fn: function that maps episode number to a dict of difficulty settings.
#FATTA A FINE FILE
class CurriculumWrapper(Wrapper):  
    def __init__(self, env, schedule_fn):
        super().__init__(env)
        self.schedule_fn = schedule_fn
        self.episode_count = 0

    def reset(self, **kwargs):
        difficulty = self.schedule_fn(self.episode_count)
        if hasattr(self.env, 'set_difficulty'):
            self.env.set_difficulty(difficulty)
        self.episode_count += 1
        return self.env.reset(**kwargs)


 #Per compoore env, DA CHIAMARE SU FILE DI TRAINIG
def make_curriculum_env(schedule_fn):
    env = HopperEnv()
    env = SlopeWrapper(env)
    env = WindWrapper(env)
    env = LegWeightWrapper(env)
    env = CurriculumWrapper(env, schedule_fn)
    return env


#Da gestire il vettore (scheduler) delle difficoltà nel file di training
#Un esempio può essere
def difficulty_scheduler(episode):
    return {
        'slope': min(0.3, episode * 0.01),
        'wind': min(15.0, episode * 0.5),
        'extra_mass': min(3.0, episode * 0.1),
    }
#In questo modo la difficoltà è progressiva
#Da usare CurriculumWrapper per applicarla

