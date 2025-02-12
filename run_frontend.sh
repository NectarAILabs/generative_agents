#!/bin/bash

FRONTEND_SCRIPT_PATH="environment/frontend_server"
FRONTEND_SCRIPT_FILE="manage.py"
CONDA_ENV="simulacra"
CONDA_PATH="/opt/homebrew/Caskroom/miniconda/base"

FILE_NAME="Bash-Script-Frontend"
echo "(${FILE_NAME}): Running frontend server"

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

cd ${FRONTEND_SCRIPT_PATH}
source "${CONDA_PATH}/etc/profile.d/conda.sh" || {
    echo "Failed to source conda.sh. Please check your conda path."
    exit 1
}
conda activate "${CONDA_ENV}" || {
    echo "Failed to activate conda environment. Please check the environment name."
    exit 1
}

PORT=8000
if [ -z "$1" ]
then
    echo "(${FILE_NAME}): No port provided. Using default port: ${PORT}"
else
    PORT=$1
fi

python3 ${FRONTEND_SCRIPT_FILE} runserver ${PORT}