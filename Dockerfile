# Use Python 3.12 image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /workspace/FantasyFootballAgent

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
#RUN python3 -m venv /venv && /venv/bin/pip install --upgrade pip && /venv/bin/pip install -r requirements.txt

# Set the default shell to use the venv
CMD ["/bin/bash"]