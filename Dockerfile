FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y \
    ffmpeg \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["/bin/bash"]
