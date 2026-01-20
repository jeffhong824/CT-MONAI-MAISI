#!/bin/bash
# 使用指定 GPU 訓練的腳本範例
# 用法: ./train_vae_ct_gpu.sh 或 GPU_IDS="0,1,2,3" ./train_vae_ct_gpu.sh

# 指定要使用的 GPU ID（用逗號分隔，不要有空格）
# 例如: GPU_IDS="0,1,2,3" 使用 GPU 0, 1, 2, 3
# 例如: GPU_IDS="0,2,4,6" 使用 GPU 0, 2, 4, 6
GPU_IDS=${GPU_IDS:-"5,6,7"}

echo "=========================================="
echo "VAE Training with Specified GPUs"
echo "GPU IDs: $GPU_IDS"
echo "=========================================="

# 設定工作目錄
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR

# 初始化 conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monai

# 設定環境變數
export PYTHONUNBUFFERED=1
export MASTER_PORT=12355
export MASTER_ADDR=localhost
export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 計算 GPU 數量
GPU_LIST=($(echo $GPU_IDS | tr ',' ' '))
NUM_GPUS=${#GPU_LIST[@]}

echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

# 執行訓練
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

echo "Training finished at $(date)"

