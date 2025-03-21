"""
Train a DQN agent on a regular gymnasium environment.
Used by app.py to train the model.
For manual training, run the following command:
    python -m src.train_scripts.train_regular <environment>
    list of available environments in config.py
"""

import os
import sys

# Environment
import gymnasium as gym  # pylint: disable=unused-import
# Environment Preprocessing
from stable_baselines3.common.env_util import make_vec_env
# Model
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.config import ROOT_PATH, regular_environments  # nopep8


def train_regular(environment, hyperparameters=None, verbose=0):
    """
    Train a DQN agent on a regular gymnasium environment.

    Args:
        environment (str): The environment that the model trains on. List of available environments in config.py
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
        str: Path to the saved model
    """
    # Set default hyperparameters
    params = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'buffer_size': 100_000,
        'batch_size': 128,
        'learning_starts': 1_000,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.1,
        'target_update_interval': 1_000,
        'train_freq': 4,
        'gradient_steps': -1,
        'n_envs': 4,
        'total_timesteps': 100_000,
        'policy_kwargs': {"net_arch": [256, 256]},
        'device': 'cuda',
        'eval_freq': 10_000,
        'n_eval_episodes': 10
    }

    # Update parameters with provided values
    if hyperparameters:
        params.update(hyperparameters)

    if environment not in regular_environments:
        raise KeyError

    if verbose:
        print(f"Starting DQN agent training on {environment} with the following parameters:")
        for key, value in params.items():
            print(f"{key}: {value}")
        print("-" * 50)

    # Create environment directory
    environment_path = ROOT_PATH / "Models" / environment
    logs_path = environment_path / "dqn_logs"
    os.makedirs(logs_path, exist_ok=True)

    # Create environment
    vec_env = make_vec_env(regular_environments[environment], n_envs=params['n_envs'], seed=0)

    # Create model
    model = DQN(
        "MlpPolicy",
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
        eval_freq=params['eval_freq'] // params['n_envs'],
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
    try:
        selected_env = sys.argv[1]
    except IndexError:
        print("Error: No environment selected")
        sys.exit(1)
    if selected_env not in regular_environments:
        print(f"Error: The environment '{selected_env}' doesn't exist.")
        print(f"Available environments: {", ".join(regular_environments.keys())}")
        sys.exit(2)

    train_regular(selected_env, verbose=1)
