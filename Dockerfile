# Use an official Miniconda runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the contents from /home/ameknas/SmartAD/ to /app inside the container
COPY . /app

# Create a new environment named 'smartad_env' and install dependencies from environment.yml
COPY environment.yml /app/environment.yml
RUN conda env update --name smartad_env --file /app/environment.yml --prune

# Make sure the environment is activated when using CMD
SHELL ["conda", "run", "-n", "smartad_env", "/bin/bash", "-c"]

# Expose the port that Flask will run on
EXPOSE 5000

# Command to run your Flask app in smartAD.py located inside /app
CMD ["conda", "run", "-n", "smartad_env", "python", "/app/smartAD.py"]