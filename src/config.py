"""
Defines important variables used in the project
"""

from pathlib import Path

# Root path of project. used to load and save files
ROOT_PATH = Path(__file__).resolve().parents[1]

# Dictionaries mapping available game names to their environment IDs
pixel_environments = {
    "breakout": "BreakoutNoFrameskip-v4",
    "pacman": "ALE/Pacman-v5",
    "donkeykong": "ALE/DonkeyKong-v5"
}
regular_environments = {
    "tetris": "tetris_gymnasium/Tetris",
    "cartpole": "CartPole-v1",
    "lunarlander": "LunarLander-v3",
    "mountaincar": "MountainCar-v0"
}
# List of all available game environments
env_list = list(pixel_environments.keys()) + list(regular_environments.keys())

# Hyperparameter details for tuning in web app
param_details = {
    'learning_rate':          {'is_float': 1, 'default': 0.001,  'min': 0.0001, 'max': 0.1,     'step': 0.0001},
    'gamma':                  {'is_float': 1, 'default': 0.99,   'min': 0.8,    'max': 1.0,     'step': 0.01},
    'exploration_fraction':   {'is_float': 1, 'default': 0.1,    'min': 0.01,   'max': 1.0,     'step': 0.01},
    'exploration_final_eps':  {'is_float': 1, 'default': 0.01,   'min': 0.001,  'max': 0.1,     'step': 0.001},
    'learning_starts':        {'is_float': 0, 'default': 1000,   'min': 100,    'max': 10_000,  'step': 100},
    'buffer_size':            {'is_float': 0, 'default': 50_000, 'min': 1_000,  'max': 100_000, 'step': 1000},
    'batch_size':             {'is_float': 0, 'default': 32,     'min': 16,     'max': 256,     'step': 16},
    'target_update_interval': {'is_float': 0, 'default': 1_000,  'min': 100,    'max': 10_000,  'step': 100},
}
