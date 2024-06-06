import os

from stable_baselines3 import DQN

from data import labels, read_wav, cqt_func, trim_CQT, TOP_N_FREQ
from train import GuitarEnv, train

env = GuitarEnv()
if not os.path.exists("dqn_guitar.zip"):
    train("dqn_guitar")
model = DQN.load("dqn_guitar")

file = input("Enter the path to the wav file: ")
while file != "":
    if os.path.exists(file) and file.endswith(".wav") and os.path.isfile(file):
        cqt_nabs, cqt_datum = cqt_func(*read_wav(file))
        trimmed_data_mean, trimmed_data_sum, TRIMMED_EQUIVALENCE = trim_CQT(
            cqt_datum, top=TOP_N_FREQ
        )
        print(trimmed_data_mean.shape)
        action, _ = model.predict(trimmed_data_mean)
        print(f"Predicted label: {labels[action]}")
    file = input("Enter the path to the wav file: ")
