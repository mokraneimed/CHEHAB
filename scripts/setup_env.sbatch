#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o logs/setup.%j.out
#SBATCH -e logs/setup.%j.err

set -eo pipefail
module purge
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate

# Dynamic user scratch directory
USER_SCRATCH="/scratch/${USER}"
PROJ_ROOT="${USER_SCRATCH}/chehab-unified"
CONDA_ENV_DIR="${USER_SCRATCH}/conda/envs/chehabEnv"

cd "$PROJ_ROOT"
mkdir -p logs

# start clean in case the interrupted install left a broken env
conda env remove -p "$CONDA_ENV_DIR" -y || true
conda env create -f environment.yml -p "$CONDA_ENV_DIR"

conda activate "$CONDA_ENV_DIR"
cd RL/pytrs
python -m pip install -e .
python -c "import pytrs; print('pytrs OK:', pytrs.__file__)"

conda clean -a -y
pip cache purge
du -sh "${USER_SCRATCH}/conda"
