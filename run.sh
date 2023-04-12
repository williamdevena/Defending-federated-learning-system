#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

echo "Starting server"
python server.py &
sleep 10  # Sleep for 3s to give the server enough time to start

# for i in `seq 1 5`; do
#     echo "Starting client$i"
#     python "Client/client$i.py" &
# done

client_names=("client1.py" "client2.py" "client3.py" "client4.py" "client5.py" "client6.py" "client7m.py" "client8m.py" "client9d.py" "client10d.py")

for client_name in "${client_names[@]}"; do
    echo "Starting $client_name"
    python "Client/$client_name" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
