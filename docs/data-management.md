# Data Management Workflow

This document details how dataset partitions, cold storage archives, and cloud training pipelines interact within this repository.

## 1. Storage Architecture

```
Local Storage (Source)
└── data/
    ├── label-origin.csv
    ├── video_metadata.csv
    └── video_parts/ (part_01 to part_13, part_14, ...)
           │
           ▼ (scripts/upload_raw_archive_to_hf.py)
Hugging Face Hub (Private Dataset: Cold Storage)
└── <username>/<dataset-repo-id>
    ├── branch: main (Canonical Dataset)
    │   ├── metadata/ (label-origin.csv, video_metadata.csv)
    │   └── videos/   (part_01.zip ... part_13.zip)
    │
    └── branch: experiment-batch-02 (Experimental / Staging)
        ├── metadata/ (experimental labels)
        └── videos/   (part_14.zip, ...)
           │
           ▼ (scripts/download_archive_from_hf.py)
Kaggle / Colab Cloud GPUs (Feature Extraction)
           │
           ▼
Kaggle Datasets (Feature Registry: Warm Storage)
├── vision-embeddings-final/ (1536-d SigLIP2 .npy)
├── asr-embeddings/ (768-d PhoBERT .npy + priors)
├── ocr-embeddings/ (768-d text .npy + priors)
└── tiktok-audio-embeddings/ (512-d CLAP .npy)
           │
           ▼
Kaggle Training Notebooks (experiments/training/)
```

---

## 2. Archiving and Uploading from Local

Use [scripts/upload_raw_archive_to_hf.py](file:///D:/Workspace/hierachical-multimodel-multilabel/scripts/upload_raw_archive_to_hf.py) to package and upload raw video parts and metadata.

### 2.1. Upload Official Dataset (Branch: `main`)

```powershell
# Upload all parts (part_01 through part_13) and metadata to the main branch
python scripts/upload_raw_archive_to_hf.py `
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" `
  --delete-local-archive-on-success
```

### 2.2. Upload Experimental Videos (Custom Branch)

When you have new experimental videos that should not pollute the canonical ground-truth dataset:

```powershell
# Upload only part_14 to a dedicated experimental branch (e.g., experiment-batch-02)
python scripts/upload_raw_archive_to_hf.py `
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" `
  --branch "experiment-batch-02" `
  --parts "part_14" `
  --skip-metadata `
  --delete-local-archive-on-success
```

* The script automatically creates the branch `experiment-batch-02` on Hugging Face if it does not exist.
* The `main` branch remains untouched.
* When experimental results are validated, you can merge the branch into `main` directly on the Hugging Face Web UI via Pull Request.

---

## 3. Downloading Datasets in Cloud Notebooks (Kaggle / Colab)

Use [scripts/download_archive_from_hf.py](file:///D:/Workspace/hierachical-multimodel-multilabel/scripts/download_archive_from_hf.py) or the programmatic Python helper.

### 3.1. Download Full Dataset (CLI with Disk-Safe Mode)

On Kaggle Notebooks, disk space in `/kaggle/working` is capped at 20 GB. Using `--disk-safe` downloads each part, extracts it, and deletes the zip file immediately:

```bash
# Download and extract all 13 parts sequentially while preserving disk space
python scripts/download_archive_from_hf.py \
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" \
  --all \
  --disk-safe
```

### 3.2. Download a Single Part (or Experimental Branch)

```bash
# Download part_01 from main
python scripts/download_archive_from_hf.py \
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" \
  --part "part_01" \
  --disk-safe

# Download an experimental part from a custom branch
python scripts/download_archive_from_hf.py \
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" \
  --branch "experiment-batch-02" \
  --part "part_14" \
  --disk-safe
```

### 3.3. Programmatic Python Snippet for Kaggle Notebooks

If you prefer calling Python code directly inside a Kaggle notebook cell without using the CLI:

```python
import os
import zipfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from kaggle_secrets import UserSecretsClient

# 1. Authenticate using Kaggle Secrets
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
repo_id = "KietTuannn/hierarchical-harmful-video-archive"
target_dir = Path("/kaggle/working/data/video_parts")
target_dir.mkdir(parents=True, exist_ok=True)

api = HfApi(token=hf_token)

# 2. Get all available part archives from Hugging Face
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision="main")
part_files = sorted([f for f in files if f.startswith("videos/") and f.endswith(".zip")])

print(f"Found {len(part_files)} video parts to download: {part_files}")

# 3. Rolling download and extraction (disk-safe)
for remote_zip in part_files:
    print(f"\nProcessing {remote_zip}...")
    local_zip = hf_hub_download(
        repo_id=repo_id,
        filename=remote_zip,
        repo_type="dataset",
        token=hf_token,
    )
    
    # Extract
    with zipfile.ZipFile(local_zip, "r") as archive:
        archive.extractall(target_dir)
    print(f"Extracted {remote_zip} to {target_dir}")
    
    # Reclaim disk space immediately
    os.remove(local_zip)
    print(f"Removed temporary zip: {local_zip}")
```

---

## 4. Pipeline Training Integration

Once multimodal modalities are extracted on Kaggle, save the outputs as Kaggle Datasets as referenced in [experiments/training/fusion_nb1_dataloader.py](file:///D:/Workspace/hierachical-multimodel-multilabel/experiments/training/fusion_nb1_dataloader.py):
- `vision-embeddings-final`
- `asr-embeddings`
- `ocr-embeddings`
- `tiktok-audio-embeddings`
