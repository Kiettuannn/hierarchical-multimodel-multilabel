"""
=============================================================================
FUSION NB-3: Stage 1 Training — Binary Classification (Harmful vs Normal)
=============================================================================
Input:
  - tiktok-data-splits/stage1_train.csv  ← filename | is_harmful
  - tiktok-data-splits/stage1_val.csv
  - (+ tất cả embedding datasets)

Output:
  /kaggle/working/stage1_checkpoints/
  ├── best_model.pt          ← checkpoint tốt nhất theo Val ROC-AUC
  ├── last_model.pt          ← checkpoint của epoch cuối
  ├── train_history.csv      ← loss, f1, roc_auc theo từng epoch
  └── stage1_run.log

Mục tiêu: Tách biệt harmful vs normal_content
Loss: BCEWithLogitsLoss với pos_weight
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install scikit-learn -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import sys, os
sys.path.append("/kaggle/input/fusion-scripts")  # Nơi chứa fusion_nb1_dataloader.py, fusion_nb2_model.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, roc_auc_score

from fusion_nb1_dataloader import make_loader
from fusion_nb2_model import FusionMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/stage1_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_CSV   = "/kaggle/input/tiktok-data-splits/stage1_train.csv"
VAL_CSV     = "/kaggle/input/tiktok-data-splits/stage1_val.csv"
OUTPUT_DIR  = Path("/kaggle/working/stage1_checkpoints")

BATCH_SIZE  = 128
LR          = 1e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 20
PATIENCE    = 5          # Early stopping
DROPOUT     = 0.3
THRESHOLD   = 0.5
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ── Cell 3: DataLoaders ───────────────────────────────────────────────────────
train_loader = make_loader(TRAIN_CSV, mode="stage1", batch_size=BATCH_SIZE, shuffle=True)
val_loader   = make_loader(VAL_CSV,   mode="stage1", batch_size=BATCH_SIZE, shuffle=False)

# Tính pos_weight: số mẫu negative / số mẫu positive
train_df   = pd.read_csv(TRAIN_CSV)
n_pos      = train_df["is_harmful"].sum()
n_neg      = len(train_df) - n_pos
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(DEVICE)
logger.info(f"Train: {len(train_df):,} | Harmful={n_pos:,} | Normal={n_neg:,}")
logger.info(f"pos_weight: {pos_weight.item():.3f}")

# ── Cell 4: Model, Loss, Optimizer ───────────────────────────────────────────
model     = FusionMLP(out_size=1, dropout=DROPOUT).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Model params: {total_params:,}")

# ── Cell 5: Train / Val functions ─────────────────────────────────────────────
def run_epoch(loader, model, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs  = []
    all_labels = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
            vision  = batch["vision"].to(device)
            asr     = batch["asr"].to(device)
            ocr     = batch["ocr"].to(device)
            clap    = batch["clap"].to(device)
            tabular = batch["tabular"].to(device)
            labels  = batch["label"].to(device)   # (B, 1)

            logits = model(vision, asr, ocr, clap, tabular)  # (B, 1)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    avg_loss = total_loss / len(loader.dataset)
    preds    = (np.array(all_probs) >= THRESHOLD).astype(int)
    f1       = f1_score(all_labels, preds, zero_division=0)
    roc_auc  = roc_auc_score(all_labels, all_probs)
    return avg_loss, f1, roc_auc

# ── Cell 6: Training loop ─────────────────────────────────────────────────────
best_roc_auc  = 0.0
patience_cnt  = 0
history       = []

logger.info("=" * 60)
logger.info("🚀 BẮT ĐẦU STAGE 1 TRAINING")
logger.info("=" * 60)

for epoch in range(1, EPOCHS + 1):
    train_loss, train_f1, train_roc = run_epoch(train_loader, model, criterion, optimizer)
    val_loss,   val_f1,   val_roc   = run_epoch(val_loader,   model, criterion)
    scheduler.step()

    logger.info(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss={train_loss:.4f} F1={train_f1:.4f} ROC={train_roc:.4f} | "
        f"Val   Loss={val_loss:.4f} F1={val_f1:.4f} ROC={val_roc:.4f}"
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss, "train_f1": train_f1, "train_roc_auc": train_roc,
        "val_loss": val_loss,     "val_f1": val_f1,     "val_roc_auc": val_roc,
    })

    # Checkpoint tốt nhất theo Val ROC-AUC
    if val_roc > best_roc_auc:
        best_roc_auc = val_roc
        patience_cnt = 0
        torch.save({
            "epoch"     : epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_roc_auc": val_roc,
            "val_f1"    : val_f1,
        }, OUTPUT_DIR / "best_model.pt")
        logger.info(f"  ✅ New best ROC-AUC: {best_roc_auc:.4f} — saved best_model.pt")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            logger.info(f"  ⏹️  Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

# Lưu checkpoint cuối
torch.save({"epoch": epoch, "model_state": model.state_dict()},
           OUTPUT_DIR / "last_model.pt")

# ── Cell 7: Save history & report ─────────────────────────────────────────────
pd.DataFrame(history).to_csv(OUTPUT_DIR / "train_history.csv", index=False)

logger.info("=" * 60)
logger.info(f"✅ Stage 1 hoàn thành! Best Val ROC-AUC: {best_roc_auc:.4f}")
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save output → Kaggle Dataset 'stage1-checkpoints'")
logger.info("  2. Attach tới fusion_nb4_stage2_train.py")
logger.info("=" * 60)
