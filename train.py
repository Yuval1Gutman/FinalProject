import gymnasium as gym  # pylint: disable=unused-import
import ale_py            # pylint: disable=unused-import
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN
# from sb3_contrib import QRDQN


# Create environment
vec_env = make_atari_env("ALE/Pacman-v5", n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

# Train agent
model = DQN(
    "CnnPolicy", vec_env,
    buffer_size=100_000,
    learning_starts=100_000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    learning_rate=1e-4,
    batch_size=32,
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    tensorboard_log="./dqn_logs/",
    device="cuda"
)
eval_callback = EvalCallback(
    vec_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=50_000,
    n_eval_episodes=15,
    deterministic=True,
    render=False
)
model.learn(total_timesteps=10_000_000, callback=eval_callback)
model.save("packman_dqn")

# Retrain agent
# model = DQN.load("breakout_dqn")
# eval_callback = EvalCallback(
#     vec_env,
#     best_model_save_path="./best_model/",
#     log_path="./logs/",
#     eval_freq=50_000,
#     n_eval_episodes=10,
#     deterministic=True,
#     render=False
# )
# model.learn(total_timesteps=1_000_000, callback=eval_callback)
# model.save("breakout_dqn")
