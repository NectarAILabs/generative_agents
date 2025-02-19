#!/bin/bash

# Check if sim_code is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <sim_code>"
    exit 1
fi
cd reverie/
# Assign the first argument to sim_code
sim_code=$1

# Run the Python script with the provided sim_code
python3 ./compress_sim_storage.py "$sim_code"