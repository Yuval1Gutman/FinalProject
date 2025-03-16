"""
Test the trained model on an Atari environment.
How to use:
    python -m src.test_scripts.test_regular <environment>
    list of available environments in config.py
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Environment
import gymnasium as gym  # pylint: disable=unused-import
# Environment Preprocessing
from stable_baselines3.common.env_util import make_vec_env
# Model
from stable_baselines3 import DQN

from src.config import ROOT_PATH, regular_environments

try:
    selected_env = sys.argv[1]
except IndexError:
    print("Error: No environment selected")
    sys.exit(1)
if selected_env not in regular_environments:
    print(f"Error: The environment '{selected_env}' doesn't exist.")
    print(f"Available environments: {", ".join(regular_environments.keys())}")
    sys.exit(1)

# Create environment
vec_env = make_vec_env(regular_environments[selected_env], n_envs=1, seed=0)

# Load model
model_path = ROOT_PATH / "Models" / selected_env / "best_model.zip"
if not model_path.exists():
    print(f"Error: The file `{model_path} doesn't exist.`")
    sys.exit(1)
model = DQN.load(model_path)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
