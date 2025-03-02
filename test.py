import gymnasium as gym  # pylint: disable=unused-import
import ale_py            # pylint: disable=unused-import
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN

# Create environment
vec_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)
# model = DQN.load("breakout_dqn")
model = DQN.load("best_model/best_model")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
