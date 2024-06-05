import os

from stable_baselines3 import DQN
from train import GuitarEnv

env = GuitarEnv()
model = DQN.load("guitar_dqn")

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

if os.path.exists(path) and path.endswith(".wav"):
    obs = # TODO: read the .wav file and convert it to a CQT graph, then trim it
    # TODO: predict!
else:
    print("Invalid path.")
    exit()