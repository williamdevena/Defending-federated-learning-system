#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

# Repeat the process 10 times
for run in `seq 1 100`; do
    echo "Run $run"

    echo "Starting server"
    python server.py &
    # server_pid=$!  # Save the server's process ID
    sleep 10  # Sleep for 10s to give the server enough time to start

    for i in `seq 1 5`; do
        echo "Starting client$i"
        python "Client/client$i.py" &
    done

    # Enable CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait

    # Stop the server
    # kill $server_pid
    sleep 5  # Sleep for 5s to give the server enough time to stop

done
