#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to run a command and log output
run_command() {
    local command="$1"
    local log_file="$2"

    echo "Starting: $command at $(date)"
    nohup python $command > "logs/$log_file" 2>&1 &

    # Get the PID of the last background process
    local pid=$!
    echo "Process ID: $pid"

    # Initialize the inactivity timer
    local inactivity_timer=0

    # Wait for the process to finish
    while kill -0 $pid 2>/dev/null; do
        echo "Process $pid is still running. Waiting... (Inactivity timer: $inactivity_timer minutes)"
        sleep 180  # Check every 3 minutes
        inactivity_timer=0  # Reset the timer as the process is still active
    done

    # Check the exit status
    wait $pid
    local exit_status=$?

    if [ $exit_status -ne 0 ]; then
        echo "Error occurred while running: $command (Exit status: $exit_status)"
        echo "Check logs/$log_file for details"
        return 1
    else
        echo "Finished: $command at $(date)"
        echo "Log file: logs/$log_file"
        echo "Last few lines of the log:"
        tail -n 5 "logs/$log_file"
        echo "------------------------"
    fi
}

# Array of commands to run
commands=(
    "main.py"
    "main.py model.model_name=\"EleutherAI/pythia-1.4b\""
    "main.py model.model_name=\"EleutherAI/pythia-2.8b\""
    "main.py model.model_name=\"facebook/opt-1.3b\""
    "main.py model.model_name=\"facebook/opt-2.7b\""
)

# Initialize the inactivity timer
inactivity_timer=0

# Run each command
for command in "${commands[@]}"; do
    # Generate log file name
    log_file=$(echo "$command" | sed 's/[^a-zA-Z0-9.]/_/g').log

    # Run the command
    if ! run_command "$command" "$log_file"; then
        echo "Error detected. Stopping execution."
        break
    fi

    # Increment the inactivity timer
    ((inactivity_timer++))

    # Check if no process has been running for 10 minutes
    if [ $inactivity_timer -ge 10 ]; then
        echo "No process has been running for 10 minutes. Shutting down the server..."
        sudo shutdown now
        exit 0
    fi
done

# All commands completed successfully
echo "All scripts finished successfully at $(date)"
echo "Logs are available in the 'logs' directory"
echo "No process has been running for 10 minutes. Shutting down the server..."