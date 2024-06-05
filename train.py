import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

from data import cqt_data, split_training_valid, labels

training_data, validation_data = split_training_valid(cqt_data)

SAMPLING_RATE = 12000
IS_USING_FULL_CQT = False

print(training_data.iloc[0]["CQT_DATA_FULL"])


class GuitarEnv(gym.Env):
    def __init__(self):
        self.training_data = training_data
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(
                training_data.iloc[0][
                    "CQT_DATA_FULL" if IS_USING_FULL_CQT else "CQT_DATA_TRIMMED"
                ].shape
            ),
        )
        self.action_space = gym.spaces.Discrete(len(labels))
        self.current_step = 0

    def observe(self):
        observation = self.training_data.iloc[self.current_step][
            "CQT_DATA_FULL" if IS_USING_FULL_CQT else "CQT_DATA_TRIMMED"
        ]
        return observation

    def step(self, action):
        # reward: 1 if the correct label is chosen, 0 otherwise
        reward = (
            1
            if labels[action] == self.training_data.iloc[self.current_step]["LABEL"]
            else 0
        )
        self.current_step += 1  # move to the next step

        done = self.current_step >= len(
            self.training_data
        )  # check if we've reached the end of the data

        info = {}  # placeholder for additional information. For now, it's empty

        if done:
            return None, reward, done, False, info

        return self.observe(), reward, False, False, info

    def reset(self, seed: int | None = None):
        self.current_step = 0  # reset the current step to 0
        np.random.seed(seed)  # seed the random number generator
        return self.observe(), {}

    def render(self):
        print(training_data.iloc[self.current_step])  # print the current state

    def close(self):
        pass  # no need to implement this method


def train():
    env = GuitarEnv()
    model = DQN(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("dqn_guitar")


if __name__ == "__main__":
    train()
