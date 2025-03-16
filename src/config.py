from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]

atari_environments = {
    "breakout": "BreakoutNoFrameskip-v4",
    "pacman": "ALE/Pacman-v5"
}
regular_environments = {
    "cartpole": "CartPole-v1",
    "lunarlander": "LunarLander-v3"
}
