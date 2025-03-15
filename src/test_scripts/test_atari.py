"""
Test the trained model on an Atari environment.
How to use:
    python -m src.test_scripts.test_atari <environment>
    available environments: breakout, pacman
"""
import sys

# Environment
import gymnasium as gym  # pylint: disable=unused-import
import ale_py            # pylint: disable=unused-import
# Environment Preprocessing
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
# Model
from stable_baselines3 import DQN

from src.config import ROOT_PATH, atari_environments

try:
    selected_env = sys.argv[1]
except IndexError:
    print("Error: No environment selected")
    sys.exit(1)
if selected_env not in atari_environments:
    print(f"Error: The environment '{selected_env}' doesn't exist.")
    print(f"Available environments: {", ".join(atari_environments.keys())}")
    sys.exit(1)

# Create environment
vec_env = make_atari_env(atari_environments[selected_env], n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

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
