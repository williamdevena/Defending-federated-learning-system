#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

# Repeat the process 10 times
for run in `seq 1 10`; do
    echo "Run $run"

    echo "Starting server"
    python server.py &
    # server_pid=$!  # Save the server's process ID
    sleep 10  # Sleep for 10s to give the server enough time to start

    client_names=("client1.py" "client2.py" "client3.py" "client4.py" "client5.py" "client6.py" "client7m.py" "client8m.py" "client9d.py" "client10d.py")

    for client_name in "${client_names[@]}"; do
        echo "Starting $client_name"
        python "Client/$client_name" &
    done

    # Enable CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait

    # Stop the server
    # kill $server_pid
    sleep 5  # Sleep for 5s to give the server enough time to stop

done
