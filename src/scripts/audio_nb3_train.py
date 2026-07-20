"""
=============================================================================
NB-3: AudioBranchModel Training — MLP Head trên CLAP Embeddings
=============================================================================
Input (từ NB-1, NB-2 Kaggle Dataset outputs):
  - /kaggle/input/tiktok-audio-embeddings/embedding_index.csv
  - /kaggle/input/tiktok-audio-embeddings/embeddings/{stem_name}.npy
  - /kaggle/input/YOUR_MAIN_DATASET/split_assignment.csv   ← dùng split chung
  - /kaggle/input/YOUR_MAIN_DATASET/label-origin.csv.csv

Output:
  /kaggle/working/audio_branch/
  ├── checkpoints/
  │   ├── audio_stage1_best.pt     ← Stage 1 binary (harmful vs normal)
  │   └── audio_stage2_best.pt     ← Stage 2 multi-label (7 nhãn)
  ├── logs/
  │   ├── stage1_metrics.json
  │   └── stage2_metrics.json
  └── audio_branch_config.json     ← lưu lại config để tái tạo

Kiến trúc:
  CLAP embedding (512-dim, frozen) → cache .npy
        ↓
  MLP Head: LayerNorm → Linear(512→256) → GELU → Dropout(0.3)
                      → Linear(256→128) → GELU → Dropout(0.2)
                      → Linear(128→7)   ← Stage 2 output (intermediate 128-dim lưu cho fusion)

Kaggle Setup:
  - Accelerator: GPU T4
  - Internet: OFF (không cần, model đã cache)
  - Runtime: Stage 1 ~30min + Stage 2 ~60min = ~1.5 giờ
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install scikit-learn -q  # cho metrics

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import time
import logging
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import (f1_score, roc_auc_score,
                             average_precision_score, classification_report)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/nb3_run.log")
    ]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
EMB_INDEX_CSV      = "/kaggle/input/tiktok-audio-embeddings/embedding_index.csv"
LABEL_CSV          = "/kaggle/input/YOUR_MAIN_DATASET/label-origin.csv.csv"
SPLIT_CSV          = "/kaggle/input/YOUR_MAIN_DATASET/split_assignment.csv"

OUTPUT_ROOT        = Path("/kaggle/working/audio_branch")
CKPT_DIR           = OUTPUT_ROOT / "checkpoints"
LOG_DIR            = OUTPUT_ROOT / "logs"

# Model config
EMBED_DIM          = 512    # CLAP larger_clap_general output dim
HIDDEN_DIM_1       = 256
HIDDEN_DIM_2       = 128    # Output dim trước classifier → dùng cho fusion
NUM_CLASSES_S2     = 7
DROPOUT_1          = 0.3
DROPOUT_2          = 0.2

# Stage 1 config
S1_EPOCHS          = 30
S1_LR              = 1e-3
S1_BATCH_SIZE      = 128
S1_WEIGHT_DECAY    = 1e-4

# Stage 2 config
S2_EPOCHS          = 50
S2_LR              = 5e-4
S2_BATCH_SIZE      = 64
S2_WEIGHT_DECAY    = 1e-4

SEED               = 42

# Thứ tự nhãn — phải khớp với cột trong label CSV
LABEL_COLS = [
    "information_harm",
    "sexual_harm",
    "psychological_harm",
    "hate_harassment_harm",
    "clickbait_harm",
    "addictive_harm",
    "physical_harm",
]

# Audio relevance weights (0.1–1.5) — đã tính từ label distribution
# Mục đích: suppress nhãn mà audio không mang tín hiệu (clickbait, information)
AUDIO_RELEVANCE_WEIGHTS = torch.tensor([
    0.1,   # information_harm     → không liên quan đến audio nền
    0.5,   # sexual_harm          → thấp
    1.5,   # psychological_harm   → rất cao (nhạc ma quái, gây sốc)
    1.2,   # hate_harassment_harm → khá cao
    0.2,   # clickbait_harm       → rất thấp (visual/text driven)
    1.0,   # addictive_harm       → trung bình
    1.5,   # physical_harm        → cao (tiếng súng, nổ, bạo lực)
], dtype=torch.float32)
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
for d in [CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Cell 3: Load & merge data ─────────────────────────────────────────────────
logger.info("Loading CSV files...")
emb_index_df = pd.read_csv(EMB_INDEX_CSV)
label_df     = pd.read_csv(LABEL_CSV)
split_df     = pd.read_csv(SPLIT_CSV)  # Columns: filename, split (train/val/test)

# Merge: label + split + embedding path
merged_df = (
    label_df
    .merge(split_df,     on="filename", how="inner")
    .merge(emb_index_df[["filename", "stem_name", "emb_path"]], on="filename", how="inner")
)
logger.info(f"After merge: {len(merged_df):,} samples with embeddings + labels + splits")
logger.info(f"Split distribution:\n{merged_df['split'].value_counts()}")

# Tạo is_harmful cho Stage 1
merged_df["is_harmful"] = (merged_df[LABEL_COLS].sum(axis=1) > 0).astype(int)
logger.info(f"Stage 1 balance: harmful={merged_df['is_harmful'].sum():,}, normal={len(merged_df)-merged_df['is_harmful'].sum():,}")


# ── Cell 4: Dataset classes ────────────────────────────────────────────────────
class AudioEmbeddingDataset(Dataset):
    """
    Load pre-cached CLAP embeddings (.npy) + labels.
    Không cần GPU load cho feature extraction — training cực nhanh.
    """
    def __init__(self, df: pd.DataFrame, label_cols: list[str]):
        self.df         = df.reset_index(drop=True)
        self.label_cols = label_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        emb = torch.from_numpy(np.load(row["emb_path"])).float()  # (512,)
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        is_harmful = torch.tensor(float(row["is_harmful"]))
        return emb, labels, is_harmful


def make_loaders(df: pd.DataFrame, label_cols: list, batch_size: int):
    """Tạo DataLoader cho train/val/test splits."""
    loaders = {}
    for split in ["train", "val", "test"]:
        split_df_sub = df[df["split"] == split]
        dataset = AudioEmbeddingDataset(split_df_sub, label_cols)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=2, pin_memory=True
        )
        logger.info(f"  {split}: {len(split_df_sub):,} samples → {len(loaders[split])} batches")
    return loaders


# ── Cell 5: Model architecture ────────────────────────────────────────────────
class AudioBranchStage1(nn.Module):
    """
    Stage 1: Binary classifier (harmful / normal).
    Input : CLAP embedding (512-dim)
    Output: logit (scalar) → sigmoid → P(harmful)
    """
    def __init__(self, embed_dim=EMBED_DIM, hidden1=HIDDEN_DIM_1, hidden2=HIDDEN_DIM_2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden1),
            nn.GELU(),
            nn.Dropout(DROPOUT_1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(DROPOUT_2),
        )
        self.head = nn.Linear(hidden2, 1)

    def forward(self, x):
        feat = self.encoder(x)      # (B, 128)
        return self.head(feat)      # (B, 1)

    def get_features(self, x):
        """Trích xuất 128-dim features cho fusion."""
        return self.encoder(x)


class AudioBranchStage2(nn.Module):
    """
    Stage 2: Multi-label classifier (7 harm categories).
    Warm-start từ Stage 1 encoder, thêm head mới size 7.
    Input : CLAP embedding (512-dim)
    Output: 7 logits → sigmoid → P(each_label)
    """
    def __init__(self, embed_dim=EMBED_DIM, hidden1=HIDDEN_DIM_1,
                 hidden2=HIDDEN_DIM_2, num_classes=NUM_CLASSES_S2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden1),
            nn.GELU(),
            nn.Dropout(DROPOUT_1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(DROPOUT_2),
        )
        self.head = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        feat = self.encoder(x)      # (B, 128)
        return self.head(feat)      # (B, 7)

    def get_features(self, x):
        """Trích xuất 128-dim features cho fusion."""
        return self.encoder(x)

    @classmethod
    def from_stage1(cls, stage1_model: AudioBranchStage1) -> "AudioBranchStage2":
        """
        Warm-start: copy encoder weights từ Stage 1, khởi tạo head mới.
        """
        s2 = cls()
        # Copy encoder weights
        s2.encoder.load_state_dict(stage1_model.encoder.state_dict())
        logger.info("Stage 2 warm-started từ Stage 1 encoder ✅")
        return s2


# ── Cell 6: Loss function với weighted BCELoss ────────────────────────────────
def compute_stage1_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    """
    Stage 1: pos_weight = N_normal / N_harmful
    Nếu harmful là class positive (1), pos_weight = N_neg/N_pos
    """
    train_df   = df[df["split"] == "train"]
    n_harmful  = train_df["is_harmful"].sum()
    n_normal   = len(train_df) - n_harmful
    pos_weight = torch.tensor([n_normal / n_harmful], dtype=torch.float32)
    logger.info(f"Stage 1 pos_weight: {pos_weight.item():.3f} (N_normal={n_normal}, N_harmful={n_harmful})")
    return pos_weight


def compute_stage2_pos_weights(df: pd.DataFrame, label_cols: list) -> torch.Tensor:
    """
    Stage 2: pos_weight_i = (N_total - N_i) / N_i  cho từng nhãn.
    Chỉ tính trên train split của harmful samples.

    pos_weight được nhân thêm với AUDIO_RELEVANCE_WEIGHTS:
    - Giảm weight nhãn không liên quan đến audio (clickbait, information)
    - Tăng weight nhãn liên quan (psychological, physical)
    """
    train_df  = df[(df["split"] == "train") & (df["is_harmful"] == 1)]
    n_total   = len(train_df)
    pos_w_raw = []

    logger.info("Stage 2 pos_weight per label (train harmful split):")
    for col in label_cols:
        n_pos = train_df[col].sum()
        if n_pos == 0:
            pw = 10.0  # fallback nếu nhãn vắng
        else:
            pw = (n_total - n_pos) / n_pos
        pos_w_raw.append(pw)
        logger.info(f"  {col:30s}: N={n_pos:4d}, raw_pw={pw:.2f}")

    pos_w_tensor = torch.tensor(pos_w_raw, dtype=torch.float32)

    # Nhân với audio relevance weights
    effective_w = pos_w_tensor * AUDIO_RELEVANCE_WEIGHTS
    logger.info("\nEffective weights (pos_weight × audio_relevance):")
    for i, col in enumerate(label_cols):
        logger.info(f"  {col:30s}: {effective_w[i]:.2f}")

    return effective_w


# ── Cell 7: Training & evaluation utils ───────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for emb, labels, is_harmful in loader:
        emb        = emb.to(device)
        labels     = labels.to(device)
        is_harmful = is_harmful.to(device)
        optimizer.zero_grad()
        logits = model(emb)
        # Stage 1: scalar output, Stage 2: 7-dim output
        if logits.shape[-1] == 1:
            loss = criterion(logits.squeeze(-1), is_harmful)
        else:
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, stage: int = 2):
    model.eval()
    all_logits = []
    all_labels = []
    all_harmful = []
    for emb, labels, is_harmful in loader:
        emb = emb.to(device)
        logits = model(emb).cpu()
        all_logits.append(logits)
        all_labels.append(labels)
        all_harmful.append(is_harmful)

    all_logits  = torch.cat(all_logits, dim=0)
    all_labels  = torch.cat(all_labels, dim=0)
    all_harmful = torch.cat(all_harmful, dim=0)

    if stage == 1:
        probs = torch.sigmoid(all_logits.squeeze(-1)).numpy()
        gt    = all_harmful.numpy()
        preds = (probs >= 0.5).astype(int)
        f1    = f1_score(gt, preds, zero_division=0)
        auc   = roc_auc_score(gt, probs) if len(np.unique(gt)) > 1 else 0.5
        return {"f1_binary": f1, "roc_auc": auc}
    else:
        probs = torch.sigmoid(all_logits).numpy()
        gt    = all_labels.numpy()
        preds = (probs >= 0.5).astype(int)
        # Per-label F1 + macro
        f1_macro = f1_score(gt, preds, average="macro", zero_division=0)
        f1_per   = f1_score(gt, preds, average=None, zero_division=0)
        try:
            ap_macro = average_precision_score(gt, probs, average="macro")
        except Exception:
            ap_macro = 0.0
        return {
            "f1_macro": f1_macro,
            "ap_macro": ap_macro,
            "f1_per_label": {col: float(f1_per[i]) for i, col in enumerate(LABEL_COLS)}
        }


def train_stage(model, loaders, optimizer, scheduler, criterion,
                n_epochs: int, ckpt_path: Path, stage: int, device):
    """Generic training loop cho cả Stage 1 và Stage 2."""
    best_metric = -1.0
    history     = []

    for epoch in range(1, n_epochs + 1):
        t0        = time.time()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], device, stage=stage)

        if stage == 1:
            monitor_val = val_metrics["f1_binary"]
            metric_str  = f"F1={val_metrics['f1_binary']:.4f} AUC={val_metrics['roc_auc']:.4f}"
        else:
            monitor_val = val_metrics["f1_macro"]
            metric_str  = f"F1_macro={val_metrics['f1_macro']:.4f} AP_macro={val_metrics['ap_macro']:.4f}"

        scheduler.step()
        elapsed = time.time() - t0

        logger.info(f"[S{stage}] Epoch {epoch:3d}/{n_epochs} | loss={train_loss:.4f} | val: {metric_str} | {elapsed:.1f}s")

        if monitor_val > best_metric:
            best_metric = monitor_val
            torch.save(model.state_dict(), str(ckpt_path))
            logger.info(f"  ✅ New best {monitor_val:.4f} → saved {ckpt_path.name}")

        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

    logger.info(f"[S{stage}] Training done. Best val metric: {best_metric:.4f}")
    return history, best_metric


# ── Cell 8: STAGE 1 — Binary classification ───────────────────────────────────
logger.info("\n" + "="*60)
logger.info("STAGE 1: Binary Classification (harmful vs normal)")
logger.info("="*60)

s1_loaders  = make_loaders(merged_df, LABEL_COLS, S1_BATCH_SIZE)
s1_model    = AudioBranchStage1().to(DEVICE)
s1_pos_w    = compute_stage1_pos_weight(merged_df).to(DEVICE)
s1_criterion = nn.BCEWithLogitsLoss(pos_weight=s1_pos_w)
s1_optimizer = AdamW(s1_model.parameters(), lr=S1_LR, weight_decay=S1_WEIGHT_DECAY)
s1_scheduler = CosineAnnealingLR(s1_optimizer, T_max=S1_EPOCHS, eta_min=1e-5)

s1_history, s1_best = train_stage(
    model=s1_model,
    loaders=s1_loaders,
    optimizer=s1_optimizer,
    scheduler=s1_scheduler,
    criterion=s1_criterion,
    n_epochs=S1_EPOCHS,
    ckpt_path=CKPT_DIR / "audio_stage1_best.pt",
    stage=1,
    device=DEVICE,
)

# Eval trên test
s1_model.load_state_dict(torch.load(CKPT_DIR / "audio_stage1_best.pt"))
s1_test_metrics = evaluate(s1_model, s1_loaders["test"], DEVICE, stage=1)
logger.info(f"[S1] Test metrics: {s1_test_metrics}")

with open(LOG_DIR / "stage1_metrics.json", "w") as f:
    json.dump({"history": s1_history, "test": s1_test_metrics}, f, indent=2)


# ── Cell 9: STAGE 2 — Multi-label classification ──────────────────────────────
logger.info("\n" + "="*60)
logger.info("STAGE 2: Multi-label Classification (7 harm labels)")
logger.info("="*60)

# Chỉ dùng harmful samples cho Stage 2
df_stage2 = merged_df[merged_df["is_harmful"] == 1].copy()
logger.info(f"Stage 2 dataset: {len(df_stage2):,} harmful samples")

s2_loaders  = make_loaders(df_stage2, LABEL_COLS, S2_BATCH_SIZE)
s2_pos_w    = compute_stage2_pos_weights(df_stage2, LABEL_COLS).to(DEVICE)
s2_criterion = nn.BCEWithLogitsLoss(pos_weight=s2_pos_w)

# Warm-start từ Stage 1 encoder
s2_model = AudioBranchStage2.from_stage1(s1_model).to(DEVICE)
s2_optimizer = AdamW(s2_model.parameters(), lr=S2_LR, weight_decay=S2_WEIGHT_DECAY)
s2_scheduler = CosineAnnealingLR(s2_optimizer, T_max=S2_EPOCHS, eta_min=1e-6)

s2_history, s2_best = train_stage(
    model=s2_model,
    loaders=s2_loaders,
    optimizer=s2_optimizer,
    scheduler=s2_scheduler,
    criterion=s2_criterion,
    n_epochs=S2_EPOCHS,
    ckpt_path=CKPT_DIR / "audio_stage2_best.pt",
    stage=2,
    device=DEVICE,
)

# Eval trên test
s2_model.load_state_dict(torch.load(CKPT_DIR / "audio_stage2_best.pt"))
s2_test_metrics = evaluate(s2_model, s2_loaders["test"], DEVICE, stage=2)
logger.info(f"[S2] Test F1_macro: {s2_test_metrics['f1_macro']:.4f}")
logger.info(f"[S2] Per-label F1:")
for col, f1 in s2_test_metrics["f1_per_label"].items():
    logger.info(f"  {col:30s}: {f1:.4f}")

with open(LOG_DIR / "stage2_metrics.json", "w") as f:
    json.dump({"history": s2_history, "test": s2_test_metrics}, f, indent=2)


# ── Cell 10: Save config cho reproducibility ──────────────────────────────────
config = {
    "clap_model"              : "laion/larger_clap_general",
    "embed_dim"               : EMBED_DIM,
    "hidden_dim_1"            : HIDDEN_DIM_1,
    "hidden_dim_2"            : HIDDEN_DIM_2,
    "num_classes"             : NUM_CLASSES_S2,
    "label_cols"              : LABEL_COLS,
    "audio_relevance_weights" : AUDIO_RELEVANCE_WEIGHTS.tolist(),
    "stage1": {"epochs": S1_EPOCHS, "lr": S1_LR, "batch": S1_BATCH_SIZE},
    "stage2": {"epochs": S2_EPOCHS, "lr": S2_LR, "batch": S2_BATCH_SIZE},
    "stage1_test_metrics"     : s1_test_metrics,
    "stage2_test_metrics"     : s2_test_metrics,
}
with open(OUTPUT_ROOT / "audio_branch_config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

logger.info("\n" + "="*60)
logger.info("🎉 Training hoàn tất!")
logger.info(f"  Stage 1 best val F1   : {s1_best:.4f}")
logger.info(f"  Stage 2 best val F1   : {s2_best:.4f}")
logger.info(f"  Stage 2 test F1_macro : {s2_test_metrics['f1_macro']:.4f}")
logger.info(f"  Checkpoints saved to  : {CKPT_DIR}")
logger.info("")
logger.info("📦 NEXT STEPS (Fusion integration):")
logger.info("  - Load audio_stage2_best.pt")
logger.info("  - Dùng model.get_features(embedding) → 128-dim vector")
logger.info("  - Concat với vision/text features trong CLIPGateFusionV5")
logger.info("="*60)
