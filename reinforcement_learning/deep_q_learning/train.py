#!/usr/bin/env python3
"""Train a DQN agent to play Atari Breakout."""

import gymnasium as gym
import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy


WINDOW_LENGTH = 4


class LegacyAPIWrapper(gym.Wrapper):
    """Convert the Gymnasium API to the API expected by keras-rl2."""

    def reset(self, **kwargs):
        """Reset the environment and return only the observation."""
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """Take an action and combine termination and truncation flags."""
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode="human"):
        """Render using the mode selected when the environment was made."""
        return self.env.render()


class AtariProcessor(Processor):
    """Normalize Atari observations and clip rewards."""

    def process_state_batch(self, batch):
        """Convert a batch of image states to normalized floats."""
        return batch.astype(np.float32) / 255.0

    def process_reward(self, reward):
        """Clip rewards to the range used by the original DQN algorithm."""
        return np.clip(reward, -1.0, 1.0)


def make_env(render_mode=None):
    """Create a preprocessed Breakout environment with the legacy API."""
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=1,
        render_mode=render_mode
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False
    )
    return LegacyAPIWrapper(env)


def build_model(observation_shape, actions):
    """Build the convolutional policy network used by the DQN agent."""
    model = Sequential()
    model.add(Permute(
        (2, 3, 1),
        input_shape=(WINDOW_LENGTH,) + observation_shape
    ))
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation="relu"))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    """Create and compile the training DQN agent."""
    memory = SequentialMemory(
        limit=1000000,
        window_length=WINDOW_LENGTH
    )
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=1000000
    )
    agent = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        processor=AtariProcessor(),
        policy=policy,
        nb_steps_warmup=50000,
        gamma=0.99,
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0,
        enable_double_dqn=True
    )
    agent.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    return agent


def main():
    """Train the Breakout agent and save its final policy weights."""
    env = make_env()
    actions = env.action_space.n
    model = build_model(env.observation_space.shape, actions)
    agent = build_agent(model, actions)

    agent.fit(
        env,
        nb_steps=1750000,
        visualize=False,
        verbose=2
    )
    agent.save_weights("policy.h5", overwrite=True)
    env.close()


if __name__ == "__main__":
    main()
