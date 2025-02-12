#!/bin/bash

BACKEND_SCRIPT_PATH="reverie/backend_server"
BACKEND_SCRIPT_FILE="reverie.py"
CONDA_ENV="simulacra"
CONDA_PATH="/opt/homebrew/Caskroom/miniconda/base"
LOGS_PATH="../../logs"

# Parse conda-specific arguments first
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda_path)
            CONDA_PATH="${2}"
            shift 2
            ;;
        --env_name)
            CONDA_ENV="${2}"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

cd ${BACKEND_SCRIPT_PATH}

source "${CONDA_PATH}/etc/profile.d/conda.sh" || {
    echo "Failed to source conda.sh. Please check your conda path."
    exit 1
}
conda activate "${CONDA_ENV}" || {
    echo "Failed to activate conda environment. Please check the environment name."
    exit 1
}

echo "Running backend server at: http://127.0.0.1:8000/simulator_home"
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
echo "Timestamp: ${timestamp}"
mkdir -p ${LOGS_PATH}
python3 ${BACKEND_SCRIPT_FILE} --origin ${1} --target ${2} 2>&1 | tee ${LOGS_PATH}/${2}_${timestamp}.txt