"""
Train a DQN agent on a Pixel/Regular environment.
Used by app.py to train the model.
For manual training, run the following command:
    python src/train_agent.py <environment>
    list of available environments in config.py
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
from stable_baselines3.common.callbacks import EvalCallback
import torch

from config import ROOT_PATH, env_list, pixel_environments, regular_environments


def train_agent(environment, hyperparameters=None, verbose=0):
    """
    Train a DQN agent on a Pixel/Regular game.

    Args:
        environment (str): The game that the model trains on. List of available games in config.py
        hyperparameters (dict): Dictionary containing hyperparameters for training.
            Supported parameters:
            - learning_rate (float): Learning rate for the optimizer
            - gamma (float): Discount factor
            - buffer_size (int): Size of the replay buffer
            - batch_size (int): Batch size for training
            - learning_starts (int): Number of steps before starting to learn
            - exploration_fraction (float): Fraction of total timesteps for exploration
            - exploration_final_eps (float): Final value of epsilon for exploration
            - target_update_interval (int): Update frequency for target network
            - train_freq (int): Number of steps between updates
            - gradient_steps (int): Number of gradient steps per update
            - n_envs (int): Number of parallel environments
            - total_timesteps (int): Total number of timesteps to train
            - policy_kwargs (dict): Additional policy network parameters
            - device (str): Device to use ('cpu' or 'cuda')

    Returns:
        None: None
    """
    # Set default hyperparameters
    params = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'buffer_size': 100_000,
        'batch_size': 32,
        'learning_starts': 100_000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.01,
        'target_update_interval': 1_000,
        'train_freq': 4,
        'gradient_steps': 1,
        'n_envs': 4,
        'total_timesteps': 10_000_000,
        'policy_kwargs': {"net_arch": [256, 256]},
        'device': 'cuda',
        'eval_freq': 50_000,
        'n_eval_episodes': 10
    }

    # Update parameters with provided values
    if hyperparameters:
        params.update(hyperparameters)

    # Create environment and preprocess based on environment type
    if environment in pixel_environments:
        vec_env = make_atari_env(pixel_environments[environment], n_envs=params['n_envs'], seed=0)
        vec_env = VecFrameStack(vec_env, n_stack=4)
        params['policy'] = "CnnPolicy"
    elif environment in regular_environments:
        vec_env = make_vec_env(regular_environments[environment], n_envs=params['n_envs'], seed=0)
        params['policy'] = "MultiInputPolicy"
    else:
        raise KeyError

    # Create environment directory
    environment_path = ROOT_PATH / "Models" / environment
    logs_path = environment_path / "dqn_logs"
    os.makedirs(logs_path, exist_ok=True)

    # Fallback to CPU if CUDA isn't available
    if params['device'] == 'cuda' and not torch.cuda.is_available():
        params['device'] = 'cpu'
        if verbose:
            print("CUDA not available, using CPU")

    # Print training details
    if verbose:
        print(f"Starting DQN agent training on {environment} with the following parameters:")
        for key, value in params.items():
            print(f"{key}: {value}")
        print("-" * 50)

    # Create model
    model = DQN(
        params['policy'],
        vec_env,
        learning_rate=params['learning_rate'],
        gamma=params['gamma'],
        buffer_size=params['buffer_size'],
        batch_size=params['batch_size'],
        learning_starts=params['learning_starts'],
        exploration_fraction=params['exploration_fraction'],
        exploration_final_eps=params['exploration_final_eps'],
        target_update_interval=params['target_update_interval'],
        train_freq=params['train_freq'],
        gradient_steps=params['gradient_steps'],
        policy_kwargs=params['policy_kwargs'],
        verbose=verbose,
        tensorboard_log=logs_path,
        device=params['device']
    )

    # Setup evaluation callback
    eval_callback = EvalCallback(
        vec_env,
        best_model_save_path=environment_path,
        log_path=logs_path,
        eval_freq=params['eval_freq'],
        n_eval_episodes=params['n_eval_episodes'],
        deterministic=True,
        render=False
    )

    # Train the model
    model.learn(total_timesteps=params['total_timesteps'], callback=eval_callback)

    # Save the final model
    model_path = environment_path / f"{environment}_dqn"
    model.save(model_path)
    if verbose:
        print(f"Model saved to: {model_path}")
        print(f"TensorBoard logs saved to: {logs_path}")


if __name__ == "__main__":
    # Load input
    try:
        selected_env = sys.argv[1]
    except IndexError:
        print("Error: No environment selected")
        sys.exit(1)

    if selected_env not in env_list:
        print(f"Error: The environment '{selected_env}' doesn't exist.")
        print(f"Available environments: {", ".join(env_list)}")
        sys.exit(2)

    train_agent(selected_env, verbose=1)
