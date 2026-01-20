# 3D VAE 訓練指南 - CT 資料集適配版

本指南說明如何使用 NV-Generate-CTMR 框架訓練 3D VAE 模型，針對 CT 影像資料集進行適配。

## 專案概述

本專案基於 [NV-Generate-CTMR](https://github.com/Project-MONAI/Tutorials/tree/main/generation/NV-Generate-CTMR) 框架，針對自定義 CT 資料集進行適配，支援：
- 多 GPU 分散式訓練（DDP）
- 指定 GPU 訓練
- 自動模型保存與驗證
- TensorBoard 監控

## 資料準備

### 資料結構

已創建 JSON datalist 檔案：
- `data/datalist_ct.json` - 包含訓練、驗證、測試資料列表
  - **訓練集**：946 個檔案（72 GB）
  - **驗證集**：190 個檔案（15 GB）
  - **測試集**：190 個檔案（16 GB）

### 資料格式

- 檔案格式：`.nii.gz` (NIfTI 壓縮格式)
- 檔案命名：`CVAI-*-tar.nii.gz`
- 資料路徑：`/media/sda3/r12922188/DB_diffusion/CENC_CEfixed/`
  - `train_all/tar/` - 訓練資料
  - `val_all/tar/` - 驗證資料
  - `test_all/tar/` - 測試資料

### 重新生成 Datalist

如果需要重新生成 datalist：

```bash
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR
python3 -c "
import json
import glob
import os

base_path = '/media/sda3/r12922188/DB_diffusion/CENC_CEfixed'

train_path = os.path.join(base_path, 'train_all/tar/')
val_path = os.path.join(base_path, 'val_all/tar/')
test_path = os.path.join(base_path, 'test_all/tar/')

train_files = sorted(glob.glob(os.path.join(train_path, 'CVAI-*-tar.nii.gz')))
val_files = sorted(glob.glob(os.path.join(val_path, 'CVAI-*-tar.nii.gz')))
test_files = sorted(glob.glob(os.path.join(test_path, 'CVAI-*-tar.nii.gz')))

datalist = {
    'training': [{'image': f, 'class': 'ct'} for f in train_files],
    'validation': [{'image': f, 'class': 'ct'} for f in val_files],
    'testing': [{'image': f, 'class': 'ct'} for f in test_files]
}

os.makedirs('data', exist_ok=True)
with open('data/datalist_ct.json', 'w') as f:
    json.dump(datalist, f, indent=2)

print(f'Created datalist: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test')
"
```

## 配置檔案

### 1. 環境配置 (`configs/environment_maisi_vae_train_ct.json`)

```json
{
    "model_dir": "/media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR/models_ct/",
    "tfevent_path": "/media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR/outputs/tfevent_ct",
    "trained_autoencoder_path": null,
    "finetune": false,
    "json_data_list": "/media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR/data/datalist_ct.json"
}
```

- `model_dir`: 模型輸出目錄
- `tfevent_path`: TensorBoard 日誌路徑
- `trained_autoencoder_path`: 預訓練模型路徑（null 表示從頭訓練）
- `finetune`: 是否進行微調
- `json_data_list`: JSON datalist 路徑

### 2. 訓練配置 (`configs/config_maisi_vae_train_ct.json`)

```json
{
    "data_option": {
        "random_aug": true,
        "spacing_type": "rand_zoom",
        "spacing": null,
        "select_channel": 0
    },
    "autoencoder_train": {
        "batch_size": 1,
        "patch_size": [96, 96, 96],
        "val_batch_size": 1,
        "val_patch_size": null,
        "val_sliding_window_patch_size": [96, 96, 64],
        "lr": 1e-4,
        "perceptual_weight": 0.3,
        "kl_weight": 1e-7,
        "adv_weight": 0.1,
        "recon_loss": "l1",
        "val_interval": 10,
        "cache": 0.3,
        "amp": true,
        "n_epochs": 150
    }
}
```

**關鍵參數說明**：
- `patch_size`: [96, 96, 96] - 訓練 patch 大小（可調整為 [64, 64, 64] 以節省記憶體）
- `batch_size`: 1 - 每個 GPU 的批次大小
- `n_epochs`: 150 - 訓練輪數
- `lr`: 1e-4 - 學習率
- `val_interval`: 10 - 每 10 個 epoch 驗證一次
- `cache`: 0.3 - 資料快取比例（0-1）

### 3. 網路配置 (`configs/config_network_rflow.json`)

- 使用 Rectified Flow 版本（推理速度快 33 倍）
- 支援靈活的影像尺寸和 voxel spacing
- 不需要 body region 標註（相較於 DDPM 版本）

## 訓練方式

### 方式 1: 使用 GPU 指定腳本（推薦用於本地執行）

最簡單的方式，支援指定 GPU：

```bash
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR

# 使用預設 GPU（預設為 5,6,7，可在腳本中修改）
./train_vae_ct_gpu.sh

# 指定特定 GPU（例如使用 GPU 0, 1, 2, 3）
GPU_IDS="0,1,2,3" ./train_vae_ct_gpu.sh

# 使用其他 GPU 組合（例如 GPU 0, 2, 4, 6）
GPU_IDS="0,2,4,6" ./train_vae_ct_gpu.sh

# 只使用 2 張 GPU
GPU_IDS="0,1" ./train_vae_ct_gpu.sh
```

**修改預設 GPU**：
編輯 `train_vae_ct_gpu.sh`，修改第 8 行：
```bash
GPU_IDS=${GPU_IDS:-"5,6,7"}  # 改為您想要的預設 GPU
```

### 方式 2: 使用 SLURM 腳本（推薦用於集群）

```bash
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR
sbatch train_vae_ct.sh
```

**指定 GPU（在 SLURM 腳本中）**：
```bash
# 在提交任務前設定環境變數
export GPU_IDS="0,1,2,3"  # 使用 GPU 0, 1, 2, 3
sbatch train_vae_ct.sh

# 或使用其他 GPU 組合
export GPU_IDS="0,2,4,6"  # 使用 GPU 0, 2, 4, 6
sbatch train_vae_ct.sh
```

### 方式 3: 直接使用 torchrun（手動指定 GPU）

```bash
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR

# 設定要使用的 GPU（用逗號分隔，不要有空格）
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_PORT=12355
export MASTER_ADDR=localhost

# 執行訓練（4 GPU）
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --master_addr=localhost \
    --master_port=12355 \
    train_vae_ct.py \
    -e ./configs/environment_maisi_vae_train_ct.json \
    -c ./configs/config_network_rflow.json \
    -t ./configs/config_maisi_vae_train_ct.json \
    -g 4
```

**使用其他 GPU 組合**：
```bash
# 使用 GPU 0, 2, 4, 6
export CUDA_VISIBLE_DEVICES="0,2,4,6"
torchrun --nproc_per_node=4 ... train_vae_ct.py ... -g 4

# 只使用 2 張 GPU (0, 1)
export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nproc_per_node=2 ... train_vae_ct.py ... -g 2
```

### 方式 4: 單 GPU 訓練

```bash
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR

# 指定單張 GPU
export CUDA_VISIBLE_DEVICES="0"
python train_vae_ct.py \
    -e ./configs/environment_maisi_vae_train_ct.json \
    -c ./configs/config_network_rflow.json \
    -t ./configs/config_maisi_vae_train_ct.json \
    -g 1
```

## 訓練腳本說明

### train_vae_ct.py

主要的訓練腳本，支援：
- 多 GPU 分散式訓練（DDP）
- 自動從 JSON datalist 載入資料
- 自動模型保存與驗證
- TensorBoard 日誌記錄

**參數說明**：
- `-e, --environment-file`: 環境配置檔案
- `-c, --config-file`: 網路配置檔案
- `-t, --training-config`: 訓練配置檔案
- `-g, --gpus`: GPU 數量

### train_vae_ct_gpu.sh

簡化的 GPU 指定腳本，自動處理：
- Conda 環境啟動
- GPU 選擇與設定
- 環境變數配置
- 訓練執行

### train_vae_ct.sh

SLURM 批次腳本，適用於集群環境。

## 輸出檔案

訓練完成後，模型會保存在：
- `models_ct/autoencoder.pt` - 最新模型（每個 epoch 更新）
- `models_ct/autoencoder_best_epoch{epoch}.pt` - 最佳驗證 loss 模型（自動保存）
- `models_ct/discriminator.pt` - Discriminator 模型

TensorBoard 日誌：
- `outputs/tfevent_ct/autoencoder/` - 訓練日誌
  - `train_epoch/*` - 每個 epoch 的訓練 loss
  - `val_epoch/*` - 每個 epoch 的驗證 loss
  - `train_iter/*` - 每個 iteration 的訓練 loss
  - `val_original_image` - 驗證原始影像
  - `val_reconstructed_image` - 驗證重建影像

## 監控訓練

### TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir=/media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR/outputs/tfevent_ct

# 或使用相對路徑
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR
tensorboard --logdir=outputs/tfevent_ct
```

然後在瀏覽器中打開 `http://localhost:6006`

### 查看訓練日誌

```bash
# 查看 SLURM 輸出（如果使用 sbatch）
tail -f logs/vae_train_ct_*.out

# 查看即時訓練進度
watch -n 1 'tail -20 logs/vae_train_ct_*.out'
```

## 訓練流程說明

1. **資料載入**：從 JSON datalist 載入訓練和驗證資料
2. **資料預處理**：應用 VAE_Transform（強度範圍、spacing、augmentation）
3. **模型初始化**：創建 Autoencoder 和 Discriminator
4. **分散式設定**：如果使用多 GPU，設定 DDP
5. **訓練循環**：
   - 每個 epoch：訓練 Generator 和 Discriminator
   - 每 N 個 epoch：驗證並保存最佳模型
6. **模型保存**：自動保存最新模型和最佳模型

## 注意事項

### GPU 記憶體優化

如果遇到 OOM（Out of Memory）錯誤，可以：

1. **減小 patch_size**：
   ```json
   "patch_size": [64, 64, 64]  // 從 [96, 96, 96] 改為 [64, 64, 64]
   ```

2. **減小 batch_size**：
   ```json
   "batch_size": 1  // 已經是 1，無法再減小
   ```

3. **增加 num_splits**（在網路配置中）：
   ```json
   "autoencoder_def": {
       "num_splits": 4  // 增加此值可以減少記憶體使用
   }
   ```

4. **減小 cache_rate**：
   ```json
   "cache": 0.1  // 從 0.3 改為 0.1
   ```

### 訓練時間估算

- **資料量**：946 個訓練檔案
- **Epochs**：150
- **GPU**：3 張 RTX 3090（24GB）
- **預計時間**：數天到數週（取決於 GPU 效能和資料複雜度）

### 驗證頻率

- 預設：每 10 個 epoch 驗證一次
- 可在配置檔案中調整 `val_interval`
- 驗證會計算重建 loss、KL loss、perceptual loss

### 早停機制

- 最佳模型會自動保存（基於驗證 loss）
- 檔案命名：`autoencoder_best_epoch{epoch}.pt`
- 同時保存最新模型：`autoencoder.pt`

### GPU 選擇建議

- **記憶體充足**：使用較大 patch size [96, 96, 96] 或 [128, 128, 128]
- **記憶體有限**：使用較小 patch size [64, 64, 64]
- **多 GPU 訓練**：建議使用 2-4 張 GPU 以加速訓練

## 故障排除

### 問題 1: CUDA Out of Memory

**解決方案**：
- 減小 `patch_size`
- 減小 `batch_size`（已經是 1）
- 增加 `num_splits`
- 使用較少的 GPU

### 問題 2: 找不到資料檔案

**檢查**：
```bash
# 確認 datalist 路徑正確
cat configs/environment_maisi_vae_train_ct.json | grep json_data_list

# 確認資料檔案存在
ls /media/sda3/r12922188/DB_diffusion/CENC_CEfixed/train_all/tar/ | head -5
```

### 問題 3: DDP 初始化失敗

**解決方案**：
- 確認 `MASTER_PORT` 未被占用
- 確認所有 GPU 可見且可用
- 檢查 `CUDA_VISIBLE_DEVICES` 設定

### 問題 4: 訓練速度太慢

**優化建議**：
- 增加 GPU 數量
- 增加 `num_workers`（在 DataLoader 中）
- 增加 `cache_rate`（如果記憶體足夠）
- 使用較小的 `val_sliding_window_patch_size`

## 後續步驟

訓練完成 VAE 後，可以：

1. **生成 Latent Embeddings**：使用訓練好的 VAE 將影像編碼到 latent space
2. **訓練 Diffusion Model**：在 latent space 訓練 diffusion model
3. **訓練 ControlNet**（可選）：如果需要條件生成

詳細步驟請參考 NV-Generate-CTMR 的原始文件。

## 參考資料

- [NV-Generate-CTMR 原始文件](https://github.com/Project-MONAI/Tutorials/tree/main/generation/NV-Generate-CTMR)
- [MAISI 論文](https://arxiv.org/pdf/2409.11169)
- [MAISI-v2 論文](https://arxiv.org/pdf/2508.05772)

## 版本資訊

- **適配日期**：2025-01-20
- **資料集**：CENC_CEfixed CT 資料集
- **框架版本**：NV-Generate-CTMR (MAISI v2 - RFlow)
- **MONAI 版本**：1.6.dev2524
- **PyTorch 版本**：2.7.1+cu126
