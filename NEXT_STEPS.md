# 下一步操作指南

## ✅ 已完成

1. **Pretrained Models 已下載** (位於 `models/` 目錄)
   - ✓ `autoencoder_v1.pt` (80MB) - Pretrained VAE Autoencoder
   - ✓ `mask_generation_autoencoder.pt` (21MB)
   - ✓ `mask_generation_diffusion_unet.pt` (753MB)
   - ✓ `diff_unet_3d_rflow-ct.pt` (2.1GB) - Pretrained Diffusion UNet
   - ✓ `controlnet_3d_rflow-ct.pt` (275MB) - Pretrained ControlNet

2. **配置文件已設置**
   - ✓ `trained_autoencoder_path: models/autoencoder_v1.pt`

## 📋 下一步：生成 Latent Embeddings

使用 pretrained autoencoder 將 paired 影像（src 和 tar）編碼到 latent space。

### 方法 1: 使用 torchrun 直接執行

```bash
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR

# 設定 GPU
export CUDA_VISIBLE_DEVICES="3,7"
export MASTER_PORT=12356
export MASTER_ADDR=localhost

# 執行 embedding 生成
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=12356 \
    -m scripts.diff_model_create_training_data \
    -e ./configs/environment_maisi_diff_model_rflow-ct.json \
    -c ./configs/config_maisi_diff_model_rflow-ct.json \
    -t ./configs/config_network_rflow.json \
    -g 2
```

### 方法 2: 創建簡化腳本

如果 `create_embeddings_ct_gpu.sh` 不存在，可以創建：

```bash
cat > create_embeddings_ct_gpu.sh << 'EOF'
#!/bin/bash
GPU_IDS=${GPU_IDS:-"3,7"}

cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monai

export CUDA_VISIBLE_DEVICES=$GPU_IDS
export MASTER_PORT=12356
export MASTER_ADDR=localhost

GPU_LIST=($(echo $GPU_IDS | tr ',' ' '))
NUM_GPUS=${#GPU_LIST[@]}

python -m torch.distributed.run \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m scripts.diff_model_create_training_data \
    -e ./configs/environment_maisi_diff_model_rflow-ct.json \
    -c ./configs/config_maisi_diff_model_rflow-ct.json \
    -t ./configs/config_network_rflow.json \
    -g ${NUM_GPUS}
EOF

chmod +x create_embeddings_ct_gpu.sh
```

然後執行：
```bash
GPU_IDS="3,7" ./create_embeddings_ct_gpu.sh
```

## 📊 預期輸出

生成完成後，應該在 `embeddings_ct/` 目錄看到：

- `CVAI-XXXX-src_emb.nii.gz` (946 個檔案) - Source latent embeddings
- `CVAI-XXXX-tar_emb.nii.gz` (946 個檔案) - Target latent embeddings
- `CVAI-XXXX-src.json` (946 個檔案) - Metadata
- `CVAI-XXXX-tar.json` (946 個檔案) - Metadata

## 🔍 驗證

```bash
# 檢查生成的 embeddings
ls /media/sda3/r12922188/DB_diffusion/CENC_CEfixed/embeddings_ct/*src* | wc -l  # 應該是 946
ls /media/sda3/r12922188/DB_diffusion/CENC_CEfixed/embeddings_ct/*tar* | wc -l  # 應該是 946
```

## 🚀 完成 Embeddings 後

生成 embeddings 完成後，可以開始訓練：

```bash
# 訓練 conditional flow matching model
GPU_IDS="3,7" ./train_diff_unet_ct_gpu.sh
```

## ⚠️ 注意事項

1. **處理時間**：946 個 pairs = 1892 個檔案，可能需要數小時
2. **GPU 記憶體**：確保有足夠的 VRAM
3. **檢查配置**：確認 `json_data_list` 指向 `datalist_ct_paired.json`

