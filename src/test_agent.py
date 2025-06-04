"""
Test the trained model on a Pixel/Regular environment.
How to use:
    python src/test_agent.py <environment>
    list of available environments in src/config.py
"""

import os
import sys

# Environment
import gymnasium as gym
import ale_py
from tetris_gymnasium.envs import Tetris
# Environment Preprocessing
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
# Model
from stable_baselines3 import DQN

from config import ROOT_PATH, env_list, pixel_environments, regular_environments


def test_agent(environment):
    """
    Test a pre-trained DQN agent on a pixel/regular game.
    Args:
        environment (str): The game that the model trains on. List of available games in config.py
    Returns:
        None: None
    """
    # Create environment and preprocess based on environment type
    if environment in pixel_environments:
        vec_env = make_atari_env(pixel_environments[environment], n_envs=1, seed=0)
        vec_env = VecFrameStack(vec_env, n_stack=4)
    elif environment in regular_environments:
        vec_env = make_vec_env(regular_environments[environment], n_envs=1, seed=0)
    else:
        print(f"Error: The environment '{environment}' doesn't exist.")
        print(f"Available environments: {", ".join(env_list)}")
        sys.exit(2)

    # Load model
    model_path = ROOT_PATH / "Models" / environment / "best_model.zip"
    if not model_path.exists():
        print(f"Error: The file `{model_path} doesn't exist.`")
        sys.exit(3)
    model = DQN.load(model_path)

    # Play game infinitely
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")


if __name__ == "__main__":
    # Load input
    try:
        selected_env = sys.argv[1]
    except IndexError:
        print("Error: No environment selected")
        sys.exit(1)
    if selected_env not in env_list:
        print(f"Error: The environment '{selected_env}' doesn't exist.")
        print(f"Available environments: {', '.join(env_list)}")
        sys.exit(2)

    test_agent(selected_env)
