"""
Create a recording of the trained model on a Pixel/Regular environment.
How to use:
    python src/record_agent.py <environment>
    list of available environments in config.py
"""

import os
import sys
import time

# Environment
import gymnasium as gym
import ale_py
from tetris_gymnasium.envs import Tetris
# Environment Preprocessing
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
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

# Create video directory if it doesn't exist
video_dir = ROOT_PATH / "src" / "static" / "videos"
video_dir.mkdir(parents=True, exist_ok=True)

# Control video length and FPS
video_fps = 30
video_length_seconds = 30
video_length_steps = video_fps * video_length_seconds  # 900 steps for 30s at 30 FPS

# Set environment FPS to 30 for video recording
if hasattr(vec_env.envs[0], 'metadata'):
    vec_env.envs[0].metadata['render_fps'] = 30

# Wrap environment with video recorder
vec_env = VecVideoRecorder(
    vec_env,
    str(video_dir),
    record_video_trigger=lambda x: x == 0,
    video_length=video_length_steps,
    name_prefix=f"{selected_env}_gameplay"
)


# Load model
model_path = ROOT_PATH / "Models" / selected_env / "best_model.zip"
if not model_path.exists():
    print(f"Error: The file `{model_path} doesn't exist.`")
    sys.exit(1)
model = DQN.load(model_path)

print(f"Recording {video_length_seconds} second video of {selected_env}...")

obs = vec_env.reset()
for _ in range(video_length_steps):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    if dones:
        obs = vec_env.reset()

vec_env.close()
print(f"Video saved to {video_dir}")
