# Data Management Workflow

This document details how dataset partitions, cold storage archives, and cloud training pipelines interact within this repository.

## 1. Storage Architecture

```
Local Storage (Source)
└── data/
    ├── label-origin.csv
    ├── video_metadata.csv
    └── video_parts/ (part_01 to part_13)
           │
           ▼ (scripts/upload_raw_archive_to_hf.py)
Hugging Face Hub (Private Dataset: Cold Storage)
└── <username>/<dataset-repo-id>
    ├── metadata/
    │   ├── label-origin.csv
    │   └── video_metadata.csv
    └── videos/
        ├── part_01.zip (~2.08 GB)
        ├── ...
        └── part_13.zip (~1.31 GB)
           │
           ▼ (Kaggle Feature Extraction Notebook)
Kaggle Datasets (Feature Registry: Warm Storage)
├── vision-embeddings-final/ (1536-d SigLIP2 .npy)
├── asr-embeddings/ (768-d PhoBERT .npy + priors)
├── ocr-embeddings/ (768-d text .npy + priors)
└── tiktok-audio-embeddings/ (512-d CLAP .npy)
           │
           ▼
Kaggle Training Notebooks (experiments/training/)
```

## 2. Archiving and Uploading from Local

Use the automated script [scripts/upload_raw_archive_to_hf.py](file:///D:/Workspace/hierachical-multimodel-multilabel/scripts/upload_raw_archive_to_hf.py) to package and upload raw video parts and metadata to Hugging Face:

```bash
# Option A: If already logged in locally via huggingface-cli, HF_TOKEN is auto-detected
python scripts/upload_raw_archive_to_hf.py `
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" `
  --delete-local-archive-on-success

# Option B: Or explicitly provide your HF_TOKEN
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"
python scripts/upload_raw_archive_to_hf.py `
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" `
  --delete-local-archive-on-success
```

Key features:
- **Zero redundant compression**: MP4 files are already H.264 compressed; the script archives them using `ZIP_STORED` mode to maximize CPU throughput.
- **Resume-capable**: Automatically checks remote Hugging Face files and skips parts that were already uploaded.
- **Disk space optimization**: `--delete-local-archive-on-success` removes local temporary zip files immediately after each part is uploaded.

## 3. Consuming Raw Videos in Cloud Notebooks (Kaggle / Colab)

To download a specific `Video Part` on Kaggle without transferring the entire 27 GB dataset:

```python
import os
import zipfile
from huggingface_hub import hf_hub_download
from kaggle_secrets import UserSecretsClient

# Retrieve Hugging Face token from Kaggle Secrets
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")

# Download targeted partition
zip_path = hf_hub_download(
    repo_id="your-username/hierarchical-harmful-video-archive",
    filename="videos/part_01.zip",
    repo_type="dataset",
    token=hf_token,
)

# Extract into working directory
with zipfile.ZipFile(zip_path, "r") as archive:
    archive.extractall("/kaggle/working/data")
```

## 4. Pipeline Training Integration

Once multimodal modalities are extracted on Kaggle, save the outputs as Kaggle Datasets as referenced in [experiments/training/fusion_nb1_dataloader.py](file:///D:/Workspace/hierachical-multimodel-multilabel/experiments/training/fusion_nb1_dataloader.py):
- `vision-embeddings-final`
- `asr-embeddings`
- `ocr-embeddings`
- `tiktok-audio-embeddings`
