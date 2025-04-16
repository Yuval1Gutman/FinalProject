"""
Create a recording of the trained model on an Atari environment.
How to use:
    python -m src.test_scripts.test_atari <environment>
    list of available environments in config.py
"""

import os
import sys
import time

# Environment
import ale_py            # pylint: disable=unused-import
import gymnasium as gym  # pylint: disable=unused-import
# Environment Preprocessing
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
# Model
from stable_baselines3 import DQN

from src.config import ROOT_PATH, atari_environments
# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


try:
    selected_env = sys.argv[1]
except IndexError:
    print("Error: No environment selected")
    sys.exit(1)
if selected_env not in atari_environments:
    print(f"Error: The environment '{selected_env}' doesn't exist.")
    print(f"Available environments: {", ".join(atari_environments.keys())}")
    sys.exit(1)

# Create video directory if it doesn't exist
video_dir = ROOT_PATH / "src" / "static" / "videos"
video_dir.mkdir(parents=True, exist_ok=True)

# Create environment
vec_env = make_atari_env(atari_environments[selected_env], n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Wrap environment with video recorder
vec_env = VecVideoRecorder(
    vec_env,
    str(video_dir),
    record_video_trigger=lambda x: x == 0,
    video_length=900,  # Approx 30 seconds at 30 FPS
    name_prefix=f"{selected_env}_gameplay"
)

# Load model
model_path = ROOT_PATH / "Models" / selected_env / "best_model.zip"
if not model_path.exists():
    print(f"Error: The file `{model_path} doesn't exist.`")
    sys.exit(1)
model = DQN.load(model_path)

print(f"Recording 30 second video of {selected_env}...")
start_time = time.time()
max_duration = 30  # 30 seconds

obs = vec_env.reset()
# Record for approximately 30 seconds
while time.time() - start_time < max_duration:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    if dones:
        obs = vec_env.reset()


vec_env.close()
print(f"Video saved to {video_dir}")
