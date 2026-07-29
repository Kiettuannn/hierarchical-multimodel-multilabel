"""
=============================================================================
FUSION NB-1: Multimodal Dataset & DataLoader
=============================================================================
Mô tả:
  PyTorch Dataset load song song tất cả modality cho 1 video.
  Được import và dùng bởi fusion_nb3_stage1_train.py và fusion_nb4_stage2_train.py.

Input (attach các Kaggle Dataset sau):
  - tiktok-data-splits/          ← split_assignment.csv, stage*.csv
  - vision-embeddings-final/     ← embeddings/{filename}.npy  [1536]
  - asr-embeddings/              ← embeddings/{stem_name}.npy [768]
  - asr-embeddings/              ← asr_harm_priors.csv
  - ocr-embeddings/              ← embeddings/{filename}.npy  [768]
  - ocr-embeddings/              ← ocr_aux_priors.csv
  - tiktok-audio-embeddings/     ← embeddings/{stem_name}.npy [512]

Dims:
  vision_emb  [1536] + asr_emb [768] + ocr_emb [768] + clap_emb [512]
  + tabular   [12]  (asr_priors [7] + ocr_priors [5])
  = continuous 3584 + tabular 12

Ghi chú về filename key:
  - label-origin / split CSVs: có thể có hoặc không có .mp4
  - Vision, OCR emb dirs: filename không có .mp4
  - ASR, CLAP emb dirs: stem_name không có .mp4
  → Dataset class tự normalize về stem_name (bỏ .mp4)
=============================================================================
"""

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS CONFIG — Chỉnh sửa cho phù hợp với tên Kaggle Dataset của bạn
# ─────────────────────────────────────────────────────────────────────────────
VISION_EMB_DIR = Path("/kaggle/input/vision-embeddings-final/embeddings")
ASR_EMB_DIR    = Path("/kaggle/input/asr-embeddings/embeddings")
OCR_EMB_DIR    = Path("/kaggle/input/ocr-embeddings/embeddings")
CLAP_EMB_DIR   = Path("/kaggle/input/tiktok-audio-embeddings/embeddings")

ASR_PRIORS_CSV = "/kaggle/input/asr-embeddings/asr_harm_priors.csv"
OCR_PRIORS_CSV = "/kaggle/input/ocr-embeddings/ocr_aux_priors.csv"

# Embedding dimensions
VISION_DIM  = 1536
ASR_DIM     = 768
OCR_DIM     = 768
CLAP_DIM    = 512
TABULAR_DIM = 12   # 7 ASR priors + 5 OCR priors

ASR_PRIOR_COLS = [
    "information_harm_signal", "sexual_harm_signal", "psychological_harm_signal",
    "hate_harassment_harm_signal", "clickbait_harm_signal",
    "addictive_harm_signal", "physical_harm_signal",
]
OCR_PRIOR_COLS = [
    "aux_psychological", "aux_hate", "aux_sexual",
    "aux_addictive", "aux_clickbait",
]
HARM_COLS = [
    "information_harm", "sexual_harm", "psychological_harm",
    "hate_harassment_harm", "clickbait_harm", "addictive_harm", "physical_harm",
]


# ── Cell 2: Helper ────────────────────────────────────────────────────────────
def stem(filename: str) -> str:
    """Normalize filename: bỏ .mp4, bỏ khoảng trắng."""
    return str(filename).strip().removesuffix(".mp4")


def load_npy(path: Path, dim: int) -> np.ndarray:
    """Load .npy hoặc trả về zero vector nếu file không tồn tại."""
    if path.exists():
        return np.load(str(path)).astype(np.float32)
    return np.zeros(dim, dtype=np.float32)


# ── Cell 3: Dataset class ─────────────────────────────────────────────────────
class MultimodalDataset(Dataset):
    """
    Args:
        csv_path : path đến stage1_train.csv, stage2_train.csv, v.v.
        mode     : 'stage1' → label is_harmful [1]
                   'stage2' → label 7 harm cols [7]
    """

    def __init__(self, csv_path: str, mode: str = "stage2"):
        assert mode in ("stage1", "stage2")
        self.mode = mode

        self.df = pd.read_csv(csv_path, dtype={"filename": str})
        self.df["stem"] = self.df["filename"].apply(stem)

        # Load tabular priors vào dict để lookup nhanh
        asr_priors = pd.read_csv(ASR_PRIORS_CSV, dtype={"filename": str, "stem_name": str})
        asr_priors["stem"] = asr_priors["stem_name"].apply(stem)
        self.asr_prior_map = asr_priors.set_index("stem")[ASR_PRIOR_COLS].fillna(0).astype(np.float32)

        ocr_priors = pd.read_csv(OCR_PRIORS_CSV, dtype={"filename": str})
        ocr_priors["stem"] = ocr_priors["filename"].apply(stem)
        self.ocr_prior_map = ocr_priors.set_index("stem")[OCR_PRIOR_COLS].fillna(0).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        s    = row["stem"]

        # ── Load embeddings (zero vector nếu thiếu) ──────────────────────────
        vision_emb = load_npy(VISION_EMB_DIR / f"{s}.npy",  VISION_DIM)
        asr_emb    = load_npy(ASR_EMB_DIR    / f"{s}.npy",  ASR_DIM)
        ocr_emb    = load_npy(OCR_EMB_DIR    / f"{s}.npy",  OCR_DIM)
        clap_emb   = load_npy(CLAP_EMB_DIR   / f"{s}.npy",  CLAP_DIM)

        # ── Load tabular priors ───────────────────────────────────────────────
        asr_prior = (self.asr_prior_map.loc[s].values
                     if s in self.asr_prior_map.index
                     else np.zeros(7, dtype=np.float32))
        ocr_prior = (self.ocr_prior_map.loc[s].values
                     if s in self.ocr_prior_map.index
                     else np.zeros(5, dtype=np.float32))
        tabular = np.concatenate([asr_prior, ocr_prior])  # [12]

        # ── Label ─────────────────────────────────────────────────────────────
        if self.mode == "stage1":
            label = np.array([row["is_harmful"]], dtype=np.float32)
        else:
            label = row[HARM_COLS].values.astype(np.float32)

        return {
            "vision" : torch.from_numpy(vision_emb),
            "asr"    : torch.from_numpy(asr_emb),
            "ocr"    : torch.from_numpy(ocr_emb),
            "clap"   : torch.from_numpy(clap_emb),
            "tabular": torch.from_numpy(tabular),
            "label"  : torch.from_numpy(label),
        }


# ── Cell 4: Factory functions ─────────────────────────────────────────────────
def make_loader(csv_path: str, mode: str, batch_size: int,
                shuffle: bool = True, num_workers: int = 2) -> DataLoader:
    dataset = MultimodalDataset(csv_path, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


# ── Cell 5: Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = make_loader(
        "/kaggle/input/tiktok-data-splits/stage2_train.csv",
        mode="stage2", batch_size=32
    )
    batch = next(iter(loader))
    print("✅ DataLoader smoke test:")
    for k, v in batch.items():
        print(f"  {k:8s}: {tuple(v.shape)}  dtype={v.dtype}")
    # Expected:
    # vision  : (32, 1536)
    # asr     : (32, 768)
    # ocr     : (32, 768)
    # clap    : (32, 512)
    # tabular : (32, 12)
    # label   : (32, 7)
