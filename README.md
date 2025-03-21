# DQN Training Platform

A comprehensive web-based platform for training and visualizing Deep Q-Network (DQN) agents on various reinforcement learning environments.

![Training Process](training_process.png)

## Features

- 🎮 Train DQN agents on multiple environments:
  - Atari games: Breakout, Pacman
  - Regular environments: CartPole, LunarLander
- ⚙️ Customize hyperparameters or use optimized defaults
- 📊 Real-time training status monitoring
- 🎥 Record and replay agent performance videos
- 🐳 Docker support for easy deployment

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- [Docker](https://www.docker.com/get-started) (optional)

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FinalProject
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker

Build and run the Docker container:
```bash
docker build -t dqn-training .
docker run -it -p 5000:5000 --gpus all dqn-training
```

## Usage

1. Start the web server:
   ```bash
   python -m src.app
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Training Interface

The web interface allows you to:

1. Select an environment (Breakout, Pacman, CartPole, LunarLander)
2. Use default hyperparameters or customize:
   - Learning rate
   - Discount factor (gamma)
   - Exploration parameters
   - Buffer size
   - Batch size
   - Update intervals
3. Start and stop training with real-time status updates

![Training Interface](training_process.png)

## Examples

Check out these examples of trained agents:

<div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
  <div style="width: 48%;">
    <h3>Breakout</h3>
    <video width="100%" controls>
      <source src="src/static/videos/breakout_gameplay.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  <div style="width: 48%;">
    <h3>Pacman</h3>
    <video width="100%" controls>
      <source src="src/static/videos/pacman_gameplay.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

<div style="display: flex; justify-content: space-between;">
  <div style="width: 48%;">
    <h3>CartPole</h3>
    <video width="100%" controls>
      <source src="src/static/videos/cartpole_gameplay.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  <div style="width: 48%;">
    <h3>LunarLander</h3>
    <video width="100%" controls>
      <source src="src/static/videos/lunarlander_gameplay.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

## Project Structure

FinalProject/
├── Models/                    # Saved models and training logs
│   ├── breakout/
│   ├── pacman/
│   ├── cartpole/
│   └── lunarlander/
├── src/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── videos/            # Recorded agent performances
│   ├── templates/
│   ├── test_scripts/          # Testing and recording scripts
│   ├── train_scripts/         # Training implementations
│   ├── app.py                 # Flask web application
│   └── config.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

## Technical Details

### DQN Implementation
This project uses Stable Baselines3 for the DQN implementation, providing:

- Experience replay buffer
- Target network for stable training
- Convolutional neural networks for Atari games
- MLP networks for regular environments

### Web Interface
- Built with Flask
- Bootstrap for responsive design
- AJAX for real-time training status updates

