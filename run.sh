#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

GREEN='\033[0;32m'
NC='\033[0m'

# Repeat the process 10 times
for run in `seq 1 10`; do
    echo "Run $run"

    echo -e "${GREEN}\n\nSTARTING SERVER\n\n${NC}"
    python server/server.py &
    # server_pid=$!  # Save the server's process ID
    sleep 10  # Sleep for 10s to give the server enough time to start


    for cid in `seq 1 6`;
    do
        echo -e "${GREEN}- STARTING CLIENT $cid${NC}"
        python "client/client.py" $cid &
    done

    for cid in `seq 7 8`;
    do
        echo -e "${GREEN}- STARTING CLIENT $cid${NC}"
        python "client/client_model_poisoning.py" $cid &
    done

    for cid in `seq 9 10`;
    do
        echo -e "${GREEN}- STARTING CLIENT $cid${NC}"
        python "client/client_data_poisoning.py" $cid &
    done

    # Enable CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait

    # Stop the server
    # kill $server_pid
    sleep 5  # Sleep for 5s to give the server enough time to stop

done
