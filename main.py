import os

from stable_baselines3 import DQN
from train import GuitarEnv
from data import read_wav, labels, trim_CQT, cqt_func

env = GuitarEnv()
model = DQN.load("dqn_guitar")

obs = env.reset()

path = input(".wav file: ")

"""
Documentation for stable-baselines3
model = DQN.load("deepq_cartpole")

env = gym.make('CartPole-v1')
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""

if os.path.exists(path) and path.endswith(".wav") and os.path.isfile(path):
    wav_data, wdf, wr, sr = read_wav(path)
    cqt_datum = cqt_func(wdf, sr)
    trimmed_data = trim_CQT(cqt_datum)
    action, _states = model.predict(trimmed_data)
    print(labels[action])
else:
    print("Invalid path.")
    exit()
