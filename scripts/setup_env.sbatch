#!/bin/bash
#SBATCH -p compute
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -o /scratch/mo3135/chehab-unified/logs/setup.%j.out
#SBATCH -e /scratch/mo3135/chehab-unified/logs/setup.%j.err

set -eo pipefail
module purge
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate

cd /scratch/mo3135/chehab-unified
mkdir -p logs

# start clean in case the interrupted install left a broken env
conda env remove -p /scratch/mo3135/conda/envs/chehabEnv -y || true
conda env create -f environment.yml -p /scratch/mo3135/conda/envs/chehabEnv

conda activate /scratch/mo3135/conda/envs/chehabEnv
cd RL/pytrs
python -m pip install -e .
python -c "import pytrs; print('pytrs OK:', pytrs.__file__)"

conda clean -a -y
pip cache purge
du -sh /scratch/mo3135/conda
