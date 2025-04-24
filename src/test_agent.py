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


# Load input
try:
    selected_env = sys.argv[1]
except IndexError:
    print("Error: No environment selected")
    sys.exit(1)

# Create environment and preprocess based on environment type
if selected_env in pixel_environments:
    vec_env = make_atari_env(pixel_environments[selected_env], n_envs=1, seed=0)
    vec_env = VecFrameStack(vec_env, n_stack=4)
elif selected_env in regular_environments:
    vec_env = make_vec_env(regular_environments[selected_env], n_envs=1, seed=0)
else:
    print(f"Error: The environment '{selected_env}' doesn't exist.")
    print(f"Available environments: {", ".join(env_list)}")
    sys.exit(1)

# Load model
model_path = ROOT_PATH / "Models" / selected_env / "best_model.zip"
if not model_path.exists():
    print(f"Error: The file `{model_path} doesn't exist.`")
    sys.exit(1)
model = DQN.load(model_path)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
