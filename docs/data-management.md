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
           ▼ (scripts/upload_raw_archive_to_hf.py / upload_batch_to_hf_branch.py)
Hugging Face Hub (Private Dataset: Cold Storage)
└── <username>/<dataset-repo-id>
    ├── branch: main (Canonical Dataset: 6,285 samples)
    │   ├── metadata/ (label-origin.csv, video_metadata.csv)
    │   └── videos/   (part_01.zip ... part_13.zip)
    │
    └── branch: version-2 (Consolidated v2: 7,624 samples)
        ├── metadata/ (consolidated label-origin.csv: 7,624 rows)
        └── videos/   (part_01.zip ... part_13.zip [inherited], part_14.zip, part_15.zip, part_16.zip)
           │
           ▼ (experiments/tools/sync_hf_to_kaggle.ipynb via Google Colab)
Kaggle Datasets (Video Registry: Warm Storage)
└── kietkolat/hierarchical-harmful-video-dataset-v2 (~37 GB)
    ├── label-origin.csv
    └── part_01.zip ... part_16.zip
           │
           ▼ (Feature Extraction on Kaggle GPU)
Kaggle Datasets (Feature Registry)
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

### 2.1. Upload Official Dataset (Branch: `main`)

```powershell
# Upload all parts (part_01 through part_13) and metadata to the main branch
python scripts/upload_raw_archive_to_hf.py `
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" `
  --delete-local-archive-on-success
```

### 2.2. Upload Individual Experimental Parts (Custom Branch)

```powershell
# Upload only part_14 to a dedicated experimental branch (e.g., experiment-batch-02)
python scripts/upload_raw_archive_to_hf.py `
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" `
  --branch "experiment-batch-02" `
  --parts "part_14" `
  --skip-metadata `
  --delete-local-archive-on-success
```

### 2.3. Upload Consolidated New Video Batches (Branch: `version-2`)

When you receive a raw unpartitioned video folder (e.g., `D:\Workspace\KLTN\consolidated_dataset_30_10` with `videos/` and `dataset.csv`), use [scripts/upload_batch_to_hf_branch.py](file:///D:/Workspace/hierachical-multimodel-multilabel/scripts/upload_batch_to_hf_branch.py) to:
1. Automatically slice the clean videos into 500-item standard chunks starting from `part_14` (`part_14`, `part_15`, `part_16`).
2. Merge the new labels with `data/label-origin.csv` (totaling 7,624 samples) into `metadata/label-origin.csv` on the target branch.
3. Automatically branch off `main` so `version-2` inherits parts 01–13 via Git LFS without re-uploading existing files.
4. Keep the local workspace completely untouched and safe.

```powershell
python scripts/upload_batch_to_hf_branch.py `
  --source-dir "D:\Workspace\KLTN\consolidated_dataset_30_10" `
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" `
  --branch "version-2" `
  --start-part-num 14 `
  --delete-local-archive-on-success
```

---

## 3. Creating Kaggle Dataset V2 via Google Colab (Cloud-to-Cloud)

Because the full dataset consists of 16 zip archives (~37 GB), downloading all archives at once in a Kaggle notebook exceeds Kaggle's 20 GB disk limit.

Use Google Colab (with 107 GB free disk) using notebook [experiments/tools/sync_hf_to_kaggle.ipynb](file:///D:/Workspace/hierachical-multimodel-multilabel/experiments/tools/sync_hf_to_kaggle.ipynb):
1. Downloads all 16 parts and `label-origin.csv` from Hugging Face branch `version-2` via datacenter fiber (~3-4 minutes).
2. Automatically creates Kaggle Dataset `kietkolat/hierarchical-harmful-video-dataset-v2` using Kaggle API (~3 minutes).
3. Zero upload bandwidth used from your local machine.

---

## 4. Downloading Datasets in Cloud Notebooks (Kaggle / Colab)

Use [scripts/download_archive_from_hf.py](file:///D:/Workspace/hierachical-multimodel-multilabel/scripts/download_archive_from_hf.py) or the programmatic Python helper.

### 4.1. Download Full Dataset (CLI with Disk-Safe Mode)

```bash
# Download and extract all parts from branch version-2 while preserving disk space
python scripts/download_archive_from_hf.py \
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" \
  --branch "version-2" \
  --all \
  --disk-safe
```

### 4.2. Download a Single Part (or Experimental Branch)

```bash
# Download part_14 from branch version-2
python scripts/download_archive_from_hf.py \
  --repo-id "KietTuannn/hierarchical-harmful-video-archive" \
  --branch "version-2" \
  --part "part_14" \
  --disk-safe
```

---

## 5. Pipeline Training Integration

Once multimodal modalities are extracted on Kaggle, save the outputs as Kaggle Datasets as referenced in [experiments/training/fusion_nb1_dataloader.py](file:///D:/Workspace/hierachical-multimodel-multilabel/experiments/training/fusion_nb1_dataloader.py):
- `vision-embeddings-final`
- `asr-embeddings`
- `ocr-embeddings`
- `tiktok-audio-embeddings`
