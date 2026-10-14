#!/bin/bash

#SBATCH -p compute
#SBATCH --mem 64GB
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 7-00:00:00

#SBATCH -J morl-ram-pid-5m
#SBATCH -o logs/output.morl_ram_pid_5m.%j.txt
#SBATCH -e logs/error.morl_ram_pid_5m.%j.txt

set +u
# Activate cluster conda environment
if [ -f "/share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate" ]; then
    source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
fi
conda activate chehabEnv 2>/dev/null || conda activate chehab 2>/dev/null || true
set -u

export PYTHONUNBUFFERED=1

# Determine project root dynamically from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJ_ROOT/RL"
export PYTHONPATH="$PROJ_ROOT/RL:$PROJ_ROOT/RL/pytrs:${PYTHONPATH:-}"

mkdir -p "$PROJ_ROOT/logs"

# ═══════════════════════════════════════════════════════════════
# Task:       RAM-constrained Multi-Objective RL (Latency + Keys)
# Framework:  MORL (Multi-Objective)
# Method:     nato_sc (Safety Checkpointing + Quadratic Penalty)
# Algo:       lagrangian_pid (PID Adaptive Multiplier)
# Budgets:    1500, 2000, 2600, 1000000 (MB)
# Timesteps:  5,000,000
# Workers:    12 envs (across 16 cores)
# ═══════════════════════════════════════════════════════════════
METHOD="nato_sc"
ALGO="lagrangian_pid"
BUDGETS="1500,2000,2600,1000000"
TIMESTEPS=5000000
NUM_ENVS=12

echo "════════════════════════════════════════════"
echo "  Framework:   MORL"
echo "  Algorithm:   $ALGO"
echo "  Method:      $METHOD"
echo "  Budgets:     $BUDGETS"
echo "  Timesteps:   $TIMESTEPS"
echo "  Workers:     $NUM_ENVS"
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "════════════════════════════════════════════"

stdbuf -oL -eL python -u -m fhe_rl \
    --framework morl train \
    --timesteps $TIMESTEPS \
    --algo "$ALGO" \
    --method "$METHOD" \
    --budgets "$BUDGETS" \
    --num_envs $NUM_ENVS \
    2>&1 | tee -a "$PROJ_ROOT/logs/train_morl_ram_pid_5m_${SLURM_JOB_ID:-local}.log"
