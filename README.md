# 3D Latent Conditional Flow Matching for NCCT→CECT Translation

本專案實現 **3D Latent Conditional Flow Matching** 用於 paired NCCT→CECT（Non-contrast CT → Contrast-enhanced CT）轉換任務，基於論文 "Latent Flow Matching for Coherent 3D Virtual Contrast Enhancement in Cardiac CT"。

## 快速開始

### 1. 下載 Pretrained Autoencoder

根據論文要求，應使用 pretrained 3D medical latent space (MAISI-style autoencoder)。下載 pretrained model：

```bash
cd /media/sda3/r12922188/MONAI/tutorials/generation/NV-Generate-CTMR

# 使用 Python 下載
python3 -c "
from scripts.download_model_data import download_model_data
download_model_data('rflow-ct', '.', model_only=True)
"
```

這會下載以下文件到 `models/` 目錄：
- `models/autoencoder_v1.pt` - Pretrained VAE autoencoder
- `models/mask_generation_autoencoder.pt` - Mask generation autoencoder
- `models/mask_generation_diffusion_unet.pt` - Mask generation diffusion UNet

**注意**：如果已經有自己訓練的 autoencoder，可以跳過此步驟，但需要修改配置文件中的 `trained_autoencoder_path`。

### 2. 準備 Paired 數據

確保數據結構如下：
```
DB_diffusion/CENC_CEfixed/train_all/
├── src/          # Source (NCCT) images
│   └── CVAI-XXXX-src.nii.gz
└── tar/          # Target (CECT) images
    └── CVAI-XXXX-tar.nii.gz
```

Paired datalist 已創建：`data/datalist_ct_paired.json`

### 3. 生成 Latent Embeddings

使用 pretrained autoencoder 將 paired 影像編碼到 latent space：

```bash
# 使用預設 GPU (5,6,7)
./create_embeddings_ct_gpu.sh

# 或指定 GPU
GPU_IDS="3,7" ./create_embeddings_ct_gpu.sh
```

**輸出**：
- `embeddings_ct/CVAI-XXXX-src_emb.nii.gz` - Source latent embeddings
- `embeddings_ct/CVAI-XXXX-tar_emb.nii.gz` - Target latent embeddings

### 4. 訓練 Conditional Flow Matching Model

在 latent space 訓練模型，學習從 NCCT latent 到 CECT latent 的 deterministic transport：

```bash
# 使用預設 GPU (5,6,7)
./train_diff_unet_ct_gpu.sh

# 或指定 GPU
GPU_IDS="3,7" ./train_diff_unet_ct_gpu.sh
```

**輸出**：
- `models_ct/diff_unet_3d_rflow_ct.pt` - Trained conditional flow matching model

## 方法說明

### Conditional Flow Matching

根據論文 "Latent Flow Matching for Coherent 3D Virtual Contrast Enhancement in Cardiac CT"，本方法實現：

1. **Latent Encoding**: 使用 pretrained VAE 將 NCCT 和 CECT 編碼到 latent space
   - `z_src = E(x_NCCT)` - Source latent
   - `z_tar = E(x_CECT)` - Target latent

2. **Flow Matching Objective**: 學習從 source latent 到 target latent 的 velocity field
   - 線性插值：`z_t = (1-t) * z_src + t * z_tar`
   - 目標 velocity：`u* = z_tar - z_src`
   - 損失函數：`L = ||v_θ(z_t, t; z_src) - (z_tar - z_src)||²`

3. **Inference**: 通過 ODE 求解從 `z_src` 到 `z_tar` 的 transport
   - 只需少量步驟（通常 30 步）
   - 比傳統 diffusion 方法快 33 倍

## 實現細節與修改說明

### 核心修改概述

本專案從原始 NV-Generate-CTMR 框架適配為支持 **paired NCCT→CECT 轉換任務**，主要修改包括：

1. **Paired Data 支持**：修改數據加載和處理流程以支持 source/target 配對
2. **Conditional Flow Matching**：實現論文中的 latent-to-latent transport 方法
3. **UNet 輸入架構**：使用殘差連接保持通道數一致性

### 修改的函數與實現

#### 1. `scripts/diff_model_create_training_data.py`

##### 函數：`process_file()`

**修改前**：僅處理單一圖像，使用 `"image"` 鍵

**修改後**：支持 paired data，新增參數：
- `is_paired: bool` - 是否為配對數據
- `suffix: str` - 輸出文件名後綴（`"_src"` 或 `"_tar"`）
- `modality: str` - 模態類型（`"ct"`, `"mri"`）

**關鍵實現**：
```python
def process_file(
    filepath: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    plain_transforms: Compose,
    new_transforms: Compose,
    logger: logging.Logger,
    modality: str = "ct",
    is_paired: bool = False,
    suffix: str = ""
) -> None:
    # 確定完整圖像路徑
    if is_paired:
        image_path = filepath  # 已為完整路徑
    else:
        image_path = os.path.join(args.data_base_dir, filepath)
    
    # 構建輸出 embedding 文件名
    base_filename = os.path.basename(filepath).replace(".nii.gz", "")
    out_filename = os.path.join(
        args.embedding_base_dir, 
        base_filename + suffix + "_emb.nii.gz"
    )
    
    # 保存 metadata JSON（spacing, modality, dim）
    out_metadata_filename = out_filename + ".json"
    metadata = {
        "spacing": spacing,
        "modality": modality,
        "dim": dim,
        "new_dim": list(nda_image.shape),
    }
```

**原理**：
- 對於 paired data，source 和 target 分別處理，生成 `_src_emb.nii.gz` 和 `_tar_emb.nii.gz`
- 保存 metadata JSON 以便訓練時讀取 spacing 和 modality 信息
- 使用 `SlidingWindowInferer` 處理大體積，避免 GPU 記憶體溢出

##### 函數：`diff_model_create_training_data()`

**修改**：自動檢測 paired data 格式

**關鍵實現**：
```python
# 檢測是否為 paired data
is_paired = len(files_raw) > 0 and "source" in files_raw[0] and "target" in files_raw[0]

if is_paired:
    # 處理 paired data
    source_filepath = files_raw[_iter]["source"]
    target_filepath = files_raw[_iter]["target"]
    modality = files_raw[_iter].get("modality", "ct")
    
    # 處理 source 圖像
    process_file(source_filepath, ..., suffix="_src")
    # 處理 target 圖像
    process_file(target_filepath, ..., suffix="_tar")
else:
    # 向後兼容單一圖像格式
    process_file(filepath, ...)
```

#### 2. `scripts/diff_model_train.py`

##### 函數：`load_filenames()`

**修改**：返回 paired data 標記和文件列表

**關鍵實現**：
```python
def load_filenames(data_list_path: str) -> tuple:
    """
    Returns:
        tuple: (is_paired, filenames_list)
    """
    # 檢測 paired data
    if "source" in filenames_train[0] and "target" in filenames_train[0]:
        # 返回 paired 文件列表
        return True, [
            {
                "source": source_base + "_src_emb.nii.gz",
                "target": target_base + "_tar_emb.nii.gz"
            }
            for item in filenames_train
        ]
    else:
        # 單一圖像格式（向後兼容）
        return False, [...]
```

##### 函數：`prepare_data()`

**修改**：支持加載 paired embeddings

**關鍵實現**：
```python
def prepare_data(..., is_paired: bool = False) -> DataLoader:
    if is_paired:
        # Paired data: 加載 source 和 target
        train_transforms_list = [
            monai.transforms.LoadImaged(keys=["source", "target"]),
            monai.transforms.EnsureChannelFirstd(keys=["source", "target"]),
            ...
        ]
    else:
        # 單一圖像格式（向後兼容）
        train_transforms_list = [
            monai.transforms.LoadImaged(keys=["image"]),
            ...
        ]
```

##### 函數：`train_one_epoch()` - **核心實現**

**修改**：實現 Conditional Flow Matching 訓練邏輯

**關鍵實現**：
```python
def train_one_epoch(..., is_paired: bool = False) -> torch.Tensor:
    for train_data in train_loader:
        is_paired_batch = "source" in train_data and "target" in train_data
        
        if is_paired_batch:
            # ===== Conditional Flow Matching =====
            source_latent = train_data["source"].to(device) * scale_factor
            target_latent = train_data["target"].to(device) * scale_factor
            
            # 1. 採樣隨機時間步 t ~ U(0,1)
            if isinstance(noise_scheduler, RFlowScheduler):
                timesteps = noise_scheduler.sample_timesteps(source_latent)
                t = timesteps.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / num_train_timesteps
            else:
                t = torch.rand(source_latent.shape[0], 1, 1, 1, device=device)
                timesteps = (t * num_train_timesteps).long().squeeze()
            
            # 2. 線性插值：z_t = (1-t) * z_src + t * z_tar
            interpolated_latent = (1 - t) * source_latent + t * target_latent
            
            # 3. 目標 velocity：u* = z_tar - z_src
            target_velocity = target_latent - source_latent
            
            # 4. UNet 輸入：使用殘差連接保持通道數
            #    條件：z_t + 0.1 * z_src（而非 concat，避免通道數翻倍）
            conditioned_latent = interpolated_latent + 0.1 * source_latent
            
            # 5. UNet 預測 velocity
            unet_inputs = {
                "x": conditioned_latent,  # [B, 4, X, Y, Z]
                "timesteps": timesteps,
                "spacing_tensor": spacing_tensor,
            }
            model_output = unet(**unet_inputs)
            
            # 6. 損失函數：L1 Loss
            loss = loss_pt(model_output.float(), target_velocity.float())
        else:
            # 原始 diffusion 訓練（向後兼容）
            ...
```

**原理說明**：

1. **線性插值路徑**：
   - 在 latent space 中，從 `z_src` 到 `z_tar` 構建直線路徑
   - `z_t = (1-t) * z_src + t * z_tar`，其中 `t ∈ [0, 1]`
   - 這定義了從 source 到 target 的確定性 transport

2. **Velocity Field 學習**：
   - 目標 velocity：`u* = z_tar - z_src`（常數，不依賴 t）
   - UNet 學習預測：`v_θ(z_t, t; z_src) ≈ u*`
   - 損失函數：`L = ||v_θ(z_t, t) - (z_tar - z_src)||₁`

3. **殘差連接設計**：
   - **問題**：UNet 輸入通道數為 4（latent channels）
   - **方案**：使用 `z_t + 0.1 * z_src` 而非 `concat([z_t, z_src])`
   - **原因**：保持 4 通道輸入，同時融入 source 信息作為條件

4. **為什麼使用殘差而非 concat**：
   - Concat 會使輸入變為 8 通道，需要修改 UNet 架構
   - 殘差連接保持 4 通道，無需修改 UNet
   - 0.1 權重平衡插值 latent 和 source 條件信息

##### 函數：`calculate_scale_factor()`

**修改**：支持 paired data 的 scale factor 計算

```python
def calculate_scale_factor(..., is_paired: bool = False) -> torch.Tensor:
    check_data = first(train_loader)
    if is_paired and "source" in check_data:
        z = check_data["source"].to(device)  # 使用 source latent
    else:
        z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
```

### 配置文件修改

#### `configs/environment_maisi_diff_model_rflow-ct.json`

**關鍵修改**：
- `json_data_list`: 指向 `datalist_ct_paired.json`
- `embedding_base_dir`: 設置為 `embeddings_ct/` 目錄
- `trained_autoencoder_path`: 使用 pretrained `models/autoencoder_v1.pt`

### 向後兼容性

所有修改都保持向後兼容：
- 自動檢測數據格式（paired 或單一圖像）
- 如果檢測到 `"image"` 鍵，使用原始邏輯
- 如果檢測到 `"source"` 和 `"target"` 鍵，使用 paired 邏輯

## 配置說明

### 環境配置

主要配置文件：`configs/environment_maisi_diff_model_rflow-ct.json`

關鍵參數：
- `trained_autoencoder_path`: Pretrained autoencoder 路徑（預設：`models/autoencoder_v1.pt`）
- `json_data_list`: Paired datalist 路徑
- `embedding_base_dir`: Embeddings 輸出目錄

### 使用自己訓練的 Autoencoder

如果使用自己訓練的 autoencoder（如 `models_ct/autoencoder.pt`），需要：

1. 創建自定義配置文件，或
2. 修改 `trained_autoencoder_path` 指向您的模型

## 依賴關係

```
階段 1: VAE 訓練（可選，建議使用 pretrained）
   ↓
階段 2: 生成 Latent Embeddings
   ↓
階段 3: 訓練 Conditional Flow Matching Model
   ↓
階段 4: ControlNet 訓練（可選）
```

## 參考文獻

- **論文**: "Latent Flow Matching for Coherent 3D Virtual Contrast Enhancement in Cardiac CT"
- **框架**: [NV-Generate-CTMR](https://github.com/Project-MONAI/Tutorials/tree/main/generation/NV-Generate-CTMR)
- **Pretrained Models**: [NVIDIA NV-Generate-CT on HuggingFace](https://huggingface.co/nvidia/NV-Generate-CT)

## 注意事項

1. **Pretrained Model**: 根據論文，建議使用 pretrained autoencoder 而非從頭訓練
2. **Paired Data**: 確保 source 和 target 影像已正確配對和對齊
3. **GPU 記憶體**: 根據影像大小調整 batch size 和 patch size
4. **訓練時間**: Conditional flow matching 訓練通常需要數天到數週

## 故障排除

### 問題：找不到 pretrained model

**解決方案**：
```bash
# 確保已下載 pretrained model
python3 -c "
from scripts.download_model_data import download_model_data
download_model_data('rflow-ct', '.', model_only=True)
"

# 檢查文件是否存在
ls -lh models/autoencoder_v1.pt
```

### 問題：Embeddings 生成失敗

**檢查**：
1. VAE 模型路徑是否正確
2. 數據文件是否存在
3. GPU 記憶體是否足夠

### 問題：訓練時通道數不匹配

**解決方案**：確保使用殘差連接而非直接連接 source 和 interpolated latent（已在代碼中實現）

## 聯繫與支持

- 對於 MONAI 相關問題，請使用 [MONAI Discussions](https://github.com/Project-MONAI/MONAI/discussions)
- 對於本專案問題，請創建 Issue

