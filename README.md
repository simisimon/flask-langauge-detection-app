# Language Detection Model

A simple flask web application, which classifies user input into english, german or spanish.

### Run the app locally

    python run.py

### Build the Docker Image

    docker build -t flask-lang-detection:latest .

### Run the Docker Container

    docker run -p 5000:5000 flask-lang-detection:latest

