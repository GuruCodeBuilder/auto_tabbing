import os

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

import numpy as np
import pandas as pd
from tqdm import tqdm

from data import labels, fft_data

EP_LEN = None # set by get_training_validation_data
RAND_SEED = 0 # random seed
N_CORES = 4 # number of cores to use
SAVE_FREQ = 5 # frequency to save model
SAVE_FOLDER = "./saves"
SAVE_PATH = "ask" # save path to resume from. Leave at 'ask' to ask or blank to restart

def get_training_validation_data(td: int | float=None):
    # if td is None, then look for the smallest label and use that as the td - 1
    if td is None:
        td = min([len(i) for i in fft_data]) - 1
    
    # if td is greater than the largest label, then return an error
    if td > max([len(i) for i in fft_data]):
        raise ValueError(f"td is greater than the largest label: {td} > {max([len(i) for i in fft_data])}")
    
    training_data_arr = []
    eval_data_arr = []
    
    # if td < 1, td is a percentage
    if td < 1:
        for label in tqdm(labels, desc='Processing data into training and evaluation sets'):
            for i, data in enumerate(fft_data[labels.index(label)]):
                if i < len(fft_data[labels.index(label)]) * td:
                    training_data_arr.append((
                        label,
                        data[0],
                        data[1],
                        np.max(data[1]),
                        data[0][np.argmax(data[1])]
                    ))
                else:
                    eval_data_arr.append((
                        label,
                        data[0],
                        data[1],
                        np.max(data[1]),
                        data[0][np.argmax(data[1])]
                    ))
    else:
        for label in tqdm(labels, desc='Processing data into training and evaluation sets'):
            for i, data in enumerate(fft_data[labels.index(label)]):
                if i < td:
                    training_data_arr.append((
                        label,
                        data[0],
                        data[1],
                        np.max(data[1]),
                        data[0][np.argmax(data[1])]
                    ))
                else:
                    eval_data_arr.append((
                        label,
                        data[0],
                        data[1],
                        np.max(data[1]),
                        data[0][np.argmax(data[1])]
                    ))
    
    training_data = pd.DataFrame(training_data_arr, columns=['label', 'freq', 'mag', 'max_mag', 'max_mag_freq'], index=None)
    eval_data = pd.DataFrame(eval_data_arr, columns=['label', 'freq', 'mag', 'max_mag', 'max_mag_freq'], index=None)

    EP_LEN = len(training_data)
    
    return training_data, eval_data

training_data, eval_data = get_training_validation_data()

class GuitarTabbingModel(gym.Env):
    def __init__(self):
        super(GuitarTabbingModel, self).__init__()
        self.action_space = spaces.Discrete(len(labels))
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(training_data.columns) - 1,), dtype=np.float32)
        self._action_to_note = {i: label for i, label in enumerate(labels)}
        self.onIndex = 0
    
    def _get_obs(self):
        return training_data.iloc[self.onIndex].values[1:-2]

    def _get_info(self):
        return training_data.iloc[self.onIndex].values[-2:]

    def step(self, action):
        note = self._action_to_note[action]
        reward = 1 if note == training_data.iloc[self.onIndex].values[0] else 0
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, True, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.onIndex += 1
        return self._get_obs(), self._get_info()

def mkenv(rank):
    def _init():
        env = GuitarTabbingModel()
        env.reset(seed=RAND_SEED+rank)
        return env
    set_random_seed(RAND_SEED)
    return _init

if __name__ == "__main__":
    set_random_seed()
    env = GuitarTabbingModel()
    env.reset(seed=RAND_SEED)
    vec_env = SubprocVecEnv([mkenv(i) for i in range(N_CORES)])
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=SAVE_FOLDER, name_prefix='training_save')
    callbacks = [checkpoint_callback]
    
    SAVE_PATH = input("Resume point (leave blank to restart): ") if SAVE_PATH == 'ask' else SAVE_PATH
    if SAVE_PATH != "":
        if os.path.exists(SAVE_FOLDER + SAVE_PATH):
            print("Loading save...")
            model = PPO.load(SAVE_PATH, env=env)
            model.n_steps = EP_LEN
            model.n_envs = N_CORES
            model.rollout_buffer.buffer_size = EP_LEN
            model.rollout_buffer.n_envs = N_CORES
            model.rollout_buffer.reset()
        else:
            raise ValueError(f"Save path {SAVE_PATH} does not exist!")
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=EP_LEN // 8, batch_size=128, n_epochs=3, gamma=0.998, tensorboard_log="./log.txt")
    
    model.learn(total_timesteps=(EP_LEN)*N_CORES*5000, callback=CallbackList(callbacks))
        
