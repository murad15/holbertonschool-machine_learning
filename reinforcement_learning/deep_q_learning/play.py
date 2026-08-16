#!/usr/bin/env python3
"""Display Atari Breakout played by a trained DQN agent."""

from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
from train import AtariProcessor, WINDOW_LENGTH, build_model, make_env


def main():
    """Load the saved policy and display five played episodes."""
    env = make_env(render_mode="human")
    actions = env.action_space.n
    model = build_model(env.observation_space.shape, actions)
    memory = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)
    policy = GreedyQPolicy()

    agent = DQNAgent(
        model=model,
        nb_actions=actions,
        memory=memory,
        processor=AtariProcessor(),
        policy=policy,
        test_policy=policy,
        nb_steps_warmup=0,
        target_model_update=10000
    )
    agent.compile(Adam(learning_rate=0.00025), metrics=["mae"])
    agent.load_weights("policy.h5")
    agent.test(env, nb_episodes=5, visualize=True)
    env.close()


if __name__ == "__main__":
    main()
