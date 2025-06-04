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


def record_agent(environment, video_length_seconds=30, video_fps=30):
    """
    Create a 30 a pre-trained DQN agent on a pixel/regular game.
    Args:
        environment (str): The game that the model trains on. List of available games in config.py
        video_length_seconds (int): duration (in seconds) of video (default: 30)
        video_fps (int): amount of frames for every second of video (default: 30)
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
        print(f"Available environments: {', '.join(env_list)}")
        sys.exit(2)

    # Create video directory if it doesn't exist
    video_dir = ROOT_PATH / "src" / "static" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Control video length and FPS
    video_length_steps = video_fps * video_length_seconds  # 900 steps for 30s at 30 FPS

    # Set environment FPS for video recording
    if hasattr(vec_env.envs[0], 'metadata'):
        vec_env.envs[0].metadata['render_fps'] = video_fps

    # Wrap environment with video recorder
    vec_env = VecVideoRecorder(
        vec_env,
        str(video_dir),
        record_video_trigger=lambda x: x == 0,
        video_length=video_length_steps,
        name_prefix=f"{environment}_gameplay"
    )

    # Load model
    model_path = ROOT_PATH / "Models" / environment / "best_model.zip"
    if not model_path.exists():
        print(f"Error: The file `{model_path} doesn't exist.`")
        sys.exit(3)
    model = DQN.load(model_path)

    # Play game for defined length
    print(f"Recording {video_length_seconds} second video of {environment}...")
    obs = vec_env.reset()
    for _ in range(video_length_steps):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        if dones:
            obs = vec_env.reset()

    # Close environment
    vec_env.close()
    print(f"Video saved to {video_dir}")


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

    record_agent(selected_env, video_length_seconds=30, video_fps=30)
