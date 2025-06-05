"""
A Flask web app.
Opens a site at port 5000, visable through the domain http://localhost:5000
Used by running the file or the command `flask run`
"""

import os
import multiprocessing

from flask import Flask, render_template, request, jsonify

from train_agent import train_agent
from config import param_details, env_list


app = Flask(__name__)
training_process = None


@app.route("/")
def index():
    """
    Load the main page of the app
    """
    return render_template("index.html", environments=env_list, hyperparameters=param_details)


@app.route('/start_training', methods=['POST'])
def start_training():
    """Start a training process based on the user's selections"""
    global training_process

    # Get environment and hyperparameters from form
    environment = request.form.get('environment', 'cartpole')

    # Get number of timestamps from form
    num_timestamps = request.form.get('num_timestamps', '1000000')
    hyperparameters['total_timestamps'] = int(num_timestamps)

    # Check if hyperparameter tuning is enabled
    use_default_hyperparameters = request.form.get('hyperparameterToggle') == 'on'

    # Set hyperparameters
    if use_default_hyperparameters:
        hyperparameters = None
    else:
        # Update the hyperparameters extraction from the request
        hyperparameters = {key: request.form.get(key, value['default']) for key, value in param_details.items()}

        # Convert to appropriate types
        for param, details in param_details.items():
            if details['is_float']:
                hyperparameters[param] = float(hyperparameters[param])
            else:
                hyperparameters[param] = int(hyperparameters[param])

    # Stop existing process if any
    if training_process is not None and training_process.is_alive():
        training_process.terminate()
        training_process.join()

    # Create and start the appropriate training process
    if environment in env_list:
        training_process = multiprocessing.Process(
            target=train_agent,
            args=(environment, hyperparameters, 1)
        )
    else:
        return jsonify({"status": "error", "message": "Invalid environment"}), 400

    # Start training
    training_process.start()
    return jsonify({"status": "started"})


@app.route('/stop_training', methods=['POST'])
def stop_training():
    """Stop an existing training process"""
    global training_process

    if training_process is not None and training_process.is_alive():
        try:
            training_process.terminate()
            training_process.join(timeout=5)  # Wait up to 5 seconds for process to terminate

            # Force kill if it's still alive
            if training_process.is_alive():
                training_process.kill()

                training_process.join(timeout=1)

            training_process = None
            return jsonify({"status": "stopped"})
        except Exception as err:
            print(f"Error stopping training: {err}")
            return jsonify({"status": "error", "message": f"Error stopping training: {err}"}), 500

    return jsonify({"status": "not_running"})


@app.route('/training_status', methods=['GET'])
def training_status():
    """Check current status of training process"""
    global training_process

    # Check if training is running
    is_training = training_process is not None and training_process.is_alive()

    return jsonify({
        "is_training": is_training
    })


@app.route('/videos')
def list_videos():
    """List all available videos in the videos directory."""
    videos_dir = os.path.join(app.static_folder, 'videos')

    # Create videos directory if it doesn't exist
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)

    videos = []
    for file in os.listdir(videos_dir):
        if file.endswith(('.mp4', '.webm', '.ogg')):
            video_name = os.path.splitext(file)[0].replace('_', ' ').title()
            videos.append({'name': video_name, 'file': file})

    return jsonify(videos)


if __name__ == "__main__":
    # Start the web app
    multiprocessing.set_start_method('spawn', force=True)
    app.run(host="0.0.0.0")
