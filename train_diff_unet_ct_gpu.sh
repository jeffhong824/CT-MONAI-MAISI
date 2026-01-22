#!/bin/bash
# Train 3D Diffusion Model (Conditional Flow Matching) for paired NCCT→CECT
# Usage: ./train_diff_unet_ct_gpu.sh or GPU_IDS="0,1,2,3" ./train_diff_unet_ct_gpu.sh

# Specify GPU IDs (comma-separated, no spaces)
GPU_IDS=${GPU_IDS:-"0,1,2,3"} # Default to GPU 0,1,2,3

echo "=========================================="
echo "Training 3D Diffusion Model (Flow Matching)"
echo "GPU IDs: $GPU_IDS"
echo "=========================================="

# Set working directory
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monai

# Set environment variables
export PYTHONUNBUFFERED=1
export MASTER_PORT=12357
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# Calculate number of GPUs
GPU_LIST=($(echo $GPU_IDS | tr ',' ' '))
NUM_GPUS=${#GPU_LIST[@]}

echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# Run training
python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m scripts.diff_model_train \
    -e ./configs/environment_maisi_diff_model_rflow-ct.json \
    -c ./configs/config_maisi_diff_model_rflow-ct.json \
    -t ./configs/config_network_rflow.json \
    -g ${NUM_GPUS}

echo "Training finished at $(date)"

