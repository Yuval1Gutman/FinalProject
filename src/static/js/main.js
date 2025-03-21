document.addEventListener('DOMContentLoaded', function() {
    // Connect sliders to input fields
    function connectSliderToInput(sliderId, inputId) {
        const slider = document.getElementById(sliderId);
        const input = document.getElementById(inputId);
        
        // Update slider fill on load
        updateSliderFill(slider);
        
        slider.addEventListener('input', function() {
            input.value = this.value;
            updateSliderFill(this);
        });
        
        input.addEventListener('input', function() {
            slider.value = this.value;
            updateSliderFill(slider);
        });
    }
    
    // Update slider fill based on value
    function updateSliderFill(slider) {
        const min = parseFloat(slider.min) || 0;
        const max = parseFloat(slider.max) || 100;
        const value = parseFloat(slider.value) || 0;
        const percent = ((value - min) / (max - min)) * 100;
        slider.style.setProperty('--value-percent', percent + '%');
    }
    
    // Adjust numeric input value
    window.adjustValue = function(inputId, increment) {
        const input = document.getElementById(inputId);
        let value = parseFloat(input.value) + increment;
        
        if (value < parseFloat(input.min)) {
            value = parseFloat(input.min);
        } else if (value > parseFloat(input.max)) {
            value = parseFloat(input.max);
        }
        
        input.value = value;
        
        // Trigger input event to update slider if connected
        const event = new Event('input');
        input.dispatchEvent(event);
    }
    
    // Connect all sliders to their input fields dynamically
    const hyperparameters = [
        'learning_rate',
        'gamma',
        'exploration_fraction',
        'epsilon_end',
        'learning_starts',
        'buffer_size',
        'batch_size',
        'target_update_interval'
    ];
    
    for (const param of hyperparameters) {
        connectSliderToInput(`${param}_slider`, param);
    }
    
    // Training status polling
    function pollTrainingStatus() {
        fetch('/training_status')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Update button states
                document.getElementById('startTraining').disabled = data.is_training;
                document.getElementById('stopTraining').disabled = !data.is_training;
                
                // Update training status indicator
                const trainingStatus = document.getElementById('trainingStatus');
                if (data.is_training) {
                    trainingStatus.innerHTML = '<span class="badge bg-success">Training in Progress</span>';
                } else {
                    trainingStatus.innerHTML = '<span class="badge bg-secondary">Not Training</span>';
                }
                
                // Continue polling if training is running
                if (data.is_training) {
                    setTimeout(pollTrainingStatus, 1000);
                } else {
                    // Check again after 5 seconds even if not training
                    // This ensures UI stays updated if training is started from another tab/device
                    setTimeout(pollTrainingStatus, 5000);
                }
            })
            .catch(error => {
                console.error('Error polling training status:', error);
                setTimeout(pollTrainingStatus, 5000);
            });
    }
    
    // Initialize status polling
    pollTrainingStatus();
    
    // Start training button
    document.getElementById('startTraining').addEventListener('click', function() {
        const form = document.getElementById('configForm');
        if (!form) {
            console.error('Configuration form not found');
            return;
        }
        
        const formData = new FormData(form);
        
        // Check if hyperparameter toggle is checked correctly
        const hyperparameterToggle = document.getElementById('hyperparameterToggle');
        if (hyperparameterToggle) {
            formData.set('hyperparameterToggle', hyperparameterToggle.checked ? 'on' : 'off');
        }
        
        console.log('Starting training with config:', {
            environment: formData.get('environment'),
            hyperparameterToggle: formData.get('hyperparameterToggle')
        });
        
        fetch('/start_training', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Training started:', data);
            
            // Update training status immediately
            const trainingStatus = document.getElementById('trainingStatus');
            trainingStatus.innerHTML = '<span class="badge bg-success">Training in Progress</span>';
            
            // Start polling for status
            pollTrainingStatus();
        })
        .catch(error => {
            console.error('Error starting training:', error);
        });
    });
    
    // Stop training button
    document.getElementById('stopTraining').addEventListener('click', function() {
        // Disable the button immediately to prevent multiple clicks
        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Stopping...';
        
        fetch('/stop_training', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Training stopped:', data);
            
            // Force update UI state
            document.getElementById('startTraining').disabled = false;
            document.getElementById('stopTraining').disabled = true;
            document.getElementById('stopTraining').innerHTML = '<i class="bi bi-stop-fill"></i> Stop Training';
            
            // Update training status
            const trainingStatus = document.getElementById('trainingStatus');
            trainingStatus.innerHTML = '<span class="badge bg-secondary">Not Training</span>';
            
            // Force an immediate status check
            pollTrainingStatus();
        })
        .catch(error => {
            console.error('Error stopping training:', error);
            document.getElementById('stopTraining').disabled = false;
            document.getElementById('stopTraining').innerHTML = '<i class="bi bi-stop-fill"></i> Stop Training';
            
            // Force an immediate status check
            pollTrainingStatus();
        });
    });
    
    // Function to toggle hyperparameters section
    function toggleHyperparameters() {
        const hyperparameterToggle = document.getElementById('hyperparameterToggle');
        const hyperparameterSection = document.querySelector('.dqn-hyperparameters');

        if (hyperparameterToggle.checked) {
            hyperparameterSection.style.opacity = '0.5';
            hyperparameterSection.style.pointerEvents = 'none';
        } else {
            hyperparameterSection.style.opacity = '1';
            hyperparameterSection.style.pointerEvents = 'auto';
        }
    }
    
    // Initialize hyperparameter toggle and add event listener
    const hyperparameterToggle = document.getElementById('hyperparameterToggle');
    if (hyperparameterToggle) {
        // Set initial state based on the toggle (checked = use default hyperparameters)
        toggleHyperparameters();
        
        // Add event listener for changes
        hyperparameterToggle.addEventListener('change', toggleHyperparameters);
    }
    
    // Video Examples Tab Functionality
    const examplesTab = document.getElementById('examples-tab');
    if (examplesTab) {
        examplesTab.addEventListener('shown.bs.tab', function() {
            loadVideos();
        });
    }
    
    // Function to load video examples
    function loadVideos() {
        const videoGallery = document.getElementById('videoGallery');
        
        fetch('/videos')
            .then(response => response.json())
            .then(videos => {
                if (videos.length === 0) {
                    videoGallery.innerHTML = `
                        <div class="col-12 text-center py-5">
                            <i class="bi bi-exclamation-circle fs-1 text-muted"></i>
                            <p class="mt-3">No video examples available yet.</p>
                            <p class="text-muted">Train an agent and save videos to see them here.</p>
                        </div>
                    `;
                    return;
                }
                
                let galleryHTML = '';
                videos.forEach(video => {
                    const videoSrc = `/static/videos/${video.file}`;
                    galleryHTML += `
                        <div class="col-md-4 mb-4">
                            <div class="card video-card">
                                <div class="card-img-top video-thumbnail">
                                    <video class="thumbnail-video" src="${videoSrc}" preload="metadata" muted></video>
                                    <div class="play-overlay" data-video-src="${videoSrc}" data-video-name="${video.name}">
                                        <i class="bi bi-play-circle-fill"></i>
                                    </div>
                                </div>
                                <div class="card-body">
                                    <h5 class="card-title">${video.name}</h5>
                                    <button class="btn btn-primary btn-sm mt-2 play-video" 
                                            data-video-src="${videoSrc}">
                                        <i class="bi bi-play-fill"></i> Play
                                    </button>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                videoGallery.innerHTML = galleryHTML;
                
                // Load thumbnails from videos
                document.querySelectorAll('.thumbnail-video').forEach(video => {
                    // Set currentTime to 0.5 seconds to get a non-black thumbnail
                    video.addEventListener('loadeddata', function() {
                        this.currentTime = 0.5;
                    });
                });
                
                // Function to play video in modal
                function playVideoInModal(videoSrc, videoName) {
                    // Set up video modal
                    const videoPlayer = document.getElementById('videoPlayer');
                    videoPlayer.src = videoSrc;
                    
                    // Set modal title
                    document.getElementById('videoModalLabel').textContent = videoName || 'Video Example';
                    
                    // Show modal
                    const videoModal = new bootstrap.Modal(document.getElementById('videoModal'));
                    videoModal.show();
                    
                    // Pause video when modal is closed
                    document.getElementById('videoModal').addEventListener('hidden.bs.modal', function() {
                        videoPlayer.pause();
                    });
                }
                
                // Add event listeners to play buttons
                document.querySelectorAll('.play-video').forEach(button => {
                    button.addEventListener('click', function() {
                        const videoSrc = this.getAttribute('data-video-src');
                        const videoName = this.closest('.card').querySelector('.card-title').textContent;
                        playVideoInModal(videoSrc, videoName);
                    });
                });
                
                // Add event listeners to play overlays (the big play icon on thumbnails)
                document.querySelectorAll('.play-overlay').forEach(overlay => {
                    overlay.addEventListener('click', function() {
                        const videoSrc = this.getAttribute('data-video-src');
                        const videoName = this.getAttribute('data-video-name');
                        playVideoInModal(videoSrc, videoName);
                    });
                });
            })
            .catch(error => {
                console.error('Error loading videos:', error);
                videoGallery.innerHTML = `
                    <div class="col-12 text-center py-5">
                        <i class="bi bi-exclamation-triangle fs-1 text-danger"></i>
                        <p class="mt-3">Error loading videos.</p>
                        <p class="text-muted">Please try again later.</p>
                    </div>
                `;
            });
    }
});
