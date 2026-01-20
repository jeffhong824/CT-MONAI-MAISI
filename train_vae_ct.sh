#!/bin/bash
#SBATCH --job-name=vae_train_ct
#SBATCH --account=MST111121
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/vae_train_ct_%j.out
#SBATCH --error=logs/vae_train_ct_%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=r12922188+twcc@csie.ntu.edu.tw

#SBATCH --export=ALL

# Ensure the logs directory exists
mkdir -p logs

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate conda environment
conda activate monai

# Force Python to flush stdout/stderr immediately
export PYTHONUNBUFFERED=1

# Show some diagnostics at start
echo "Job started on $(hostname) at $(date)"
echo "Using Python: $(which python)"
nvidia-smi
echo "Using conda env: $(conda info --envs | grep monai)"

# Run the training script with torchrun for multi-GPU support
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR
export MASTER_PORT=12355
export MASTER_ADDR=localhost

# GPU selection: Use GPU_IDS environment variable or SLURM default
# Example: GPU_IDS="0,1,2,3" or GPU_IDS="0,2,4,6"
# If not set, use all GPUs from SLURM
if [ -z "$GPU_IDS" ]; then
    # Get number of GPUs from SLURM
    NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}
    echo "Using all ${NUM_GPUS} GPUs from SLURM"
else
    # Parse GPU IDs (comma-separated)
    GPU_LIST=($(echo $GPU_IDS | tr ',' ' '))
    NUM_GPUS=${#GPU_LIST[@]}
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "Using specified GPUs: $GPU_IDS (${NUM_GPUS} GPUs)"
fi

# Use Python module to run torchrun
python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_vae_ct.py \
    -e ./configs/environment_maisi_vae_train_ct.json \
    -c ./configs/config_network_rflow.json \
    -t ./configs/config_maisi_vae_train_ct.json \
    -g ${NUM_GPUS}

echo "Job finished at $(date)"

