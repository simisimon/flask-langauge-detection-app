# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Set the environment variable for Flask
ENV FLASK_APP=run.py

# Expose the port that Flask runs on
EXPOSE 8080

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]