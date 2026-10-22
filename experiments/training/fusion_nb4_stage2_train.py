"""
=============================================================================
FUSION NB-4: Stage 2 Training — Multi-label Classification (7 harm labels)
=============================================================================
Input:
  - stage1-checkpoints/best_model.pt     ← warm-start từ Stage 1
  - tiktok-data-splits/stage2_train.csv  ← filename | 7 harm cols (harmful only)
  - tiktok-data-splits/stage2_val.csv
  - (+ tất cả embedding datasets)

Output:
  /kaggle/working/stage2_checkpoints/
  ├── best_model.pt          ← checkpoint tốt nhất theo Val F1-Macro
  ├── last_model.pt
  ├── train_history.csv
  ├── best_thresholds.json   ← ngưỡng tối ưu cho từng nhãn (từ Val set)
  └── stage2_run.log

Warm-start strategy:
  - Load Stage 1 weights → replace head (1 → 7 neurons)
  - Epoch 1-2: Freeze MLP, chỉ train head mới
  - Epoch 3+: Unfreeze toàn bộ với lr nhỏ hơn
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install scikit-learn -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import sys
sys.path.append("/kaggle/input/fusion-scripts")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from fusion_nb1_dataloader import make_loader, HARM_COLS
from fusion_nb2_model import FusionMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/stage2_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
STAGE1_CKPT  = "/kaggle/input/stage1-checkpoints/best_model.pt"
TRAIN_CSV    = "/kaggle/input/tiktok-data-splits/stage2_train.csv"
VAL_CSV      = "/kaggle/input/tiktok-data-splits/stage2_val.csv"
OUTPUT_DIR   = Path("/kaggle/working/stage2_checkpoints")

BATCH_SIZE   = 64
LR_FROZEN    = 3e-4   # LR khi MLP frozen (chỉ train head)
LR_UNFROZEN  = 5e-5   # LR sau khi unfreeze toàn bộ
WEIGHT_DECAY = 1e-4
EPOCHS       = 30
FREEZE_EPOCHS= 2      # Số epoch freeze MLP
PATIENCE     = 7
DROPOUT      = 0.3
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ── Cell 3: DataLoaders ───────────────────────────────────────────────────────
train_loader = make_loader(TRAIN_CSV, mode="stage2", batch_size=BATCH_SIZE, shuffle=True)
val_loader   = make_loader(VAL_CSV,   mode="stage2", batch_size=BATCH_SIZE, shuffle=False)

# Tính pos_weight riêng cho từng nhãn trong 7 nhãn
train_df  = pd.read_csv(TRAIN_CSV)
n_total   = len(train_df)
pos_weights = []
logger.info("Per-label pos_weight (Stage 2):")
for col in HARM_COLS:
    n_pos = train_df[col].sum()
    n_neg = n_total - n_pos
    pw    = n_neg / max(n_pos, 1)
    pos_weights.append(pw)
    logger.info(f"  {col:<30}: pos={n_pos:4d} neg={n_neg:4d} pw={pw:.2f}")

pos_weight_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

# ── Cell 4: Warm-start Model ──────────────────────────────────────────────────
logger.info(f"Loading Stage 1 checkpoint: {STAGE1_CKPT}")
model = FusionMLP(out_size=1, dropout=DROPOUT).to(DEVICE)
ckpt  = torch.load(STAGE1_CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
logger.info(f"Stage 1 loaded (Val ROC-AUC was {ckpt.get('val_roc_auc', '?'):.4f})")

# Thay head từ 1 neuron → 7 neurons
model.replace_head(new_out_size=7)

# Bắt đầu với MLP frozen
model.freeze_mlp()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_FROZEN, weight_decay=WEIGHT_DECAY
)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# ── Cell 5: Metric helpers ────────────────────────────────────────────────────
def compute_f1_macro(labels: np.ndarray, probs: np.ndarray,
                     threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int)
    return f1_score(labels, preds, average="macro", zero_division=0)


def find_best_thresholds(labels: np.ndarray, probs: np.ndarray) -> list:
    """Tìm threshold tối ưu cho từng nhãn trên Val set."""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_ts = []
    for i in range(labels.shape[1]):
        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (probs[:, i] >= t).astype(int)
            f1    = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_ts.append(float(round(best_t, 2)))
    return best_ts


# ── Cell 6: Train / Val functions ─────────────────────────────────────────────
def run_epoch(loader, model, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    all_probs   = []
    all_labels  = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, desc="train" if is_train else "val", leave=False):
            vision  = batch["vision"].to(device)
            asr     = batch["asr"].to(device)
            ocr     = batch["ocr"].to(device)
            clap    = batch["clap"].to(device)
            tabular = batch["tabular"].to(device)
            labels  = batch["label"].to(device)   # (B, 7)

            logits = model(vision, asr, ocr, clap, tabular)  # (B, 7)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss   = total_loss / len(loader.dataset)
    all_probs  = np.vstack(all_probs)   # (N, 7)
    all_labels = np.vstack(all_labels)  # (N, 7)
    f1_macro   = compute_f1_macro(all_labels, all_probs)

    return avg_loss, f1_macro, all_probs, all_labels


# ── Cell 7: Training loop ─────────────────────────────────────────────────────
best_f1_macro = 0.0
patience_cnt  = 0
history       = []
scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

logger.info("=" * 60)
logger.info("🚀 BẮT ĐẦU STAGE 2 TRAINING (Warm-start từ Stage 1)")
logger.info("=" * 60)

for epoch in range(1, EPOCHS + 1):

    # Unfreeze sau FREEZE_EPOCHS epoch
    if epoch == FREEZE_EPOCHS + 1:
        model.unfreeze_all()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LR_UNFROZEN, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - FREEZE_EPOCHS
        )
        logger.info(f"  🔓 Epoch {epoch}: MLP unfrozen, LR reset to {LR_UNFROZEN}")

    train_loss, train_f1, _, _          = run_epoch(train_loader, model, criterion, optimizer)
    val_loss,   val_f1,  val_probs, val_labels = run_epoch(val_loader, model, criterion)
    scheduler.step()

    # Per-label F1 trên Val
    per_label_f1 = f1_score(val_labels, (val_probs >= 0.5).astype(int),
                             average=None, zero_division=0)
    per_label_str = " | ".join(
        f"{col.replace('_harm','')[:6]}={f1:.3f}"
        for col, f1 in zip(HARM_COLS, per_label_f1)
    )

    logger.info(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss={train_loss:.4f} F1={train_f1:.4f} | "
        f"Val Loss={val_loss:.4f} F1-Macro={val_f1:.4f}"
    )
    logger.info(f"  Per-label: {per_label_str}")

    history.append({
        "epoch": epoch,
        "train_loss": train_loss, "train_f1_macro": train_f1,
        "val_loss": val_loss,     "val_f1_macro": val_f1,
        **{f"val_f1_{col}": float(f1) for col, f1 in zip(HARM_COLS, per_label_f1)},
    })

    # Checkpoint theo Val F1-Macro
    if val_f1 > best_f1_macro:
        best_f1_macro = val_f1
        patience_cnt  = 0
        # Tìm best threshold cho từng nhãn
        best_ts = find_best_thresholds(val_labels, val_probs)
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_f1_macro": val_f1,
            "per_label_f1": per_label_f1.tolist(),
            "best_thresholds": best_ts,
        }, OUTPUT_DIR / "best_model.pt")
        logger.info(f"  ✅ New best F1-Macro: {best_f1_macro:.4f} — saved best_model.pt")
        logger.info(f"  Best thresholds: {dict(zip(HARM_COLS, best_ts))}")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            logger.info(f"  ⏹️  Early stopping at epoch {epoch}")
            break

# Lưu checkpoint cuối
torch.save({"epoch": epoch, "model_state": model.state_dict()},
           OUTPUT_DIR / "last_model.pt")

# ── Cell 8: Save history & thresholds ─────────────────────────────────────────
pd.DataFrame(history).to_csv(OUTPUT_DIR / "train_history.csv", index=False)

# Load best threshold từ best checkpoint
best_ckpt = torch.load(OUTPUT_DIR / "best_model.pt", map_location="cpu")
threshold_dict = dict(zip(HARM_COLS, best_ckpt["best_thresholds"]))
with open(OUTPUT_DIR / "best_thresholds.json", "w") as f:
    json.dump(threshold_dict, f, indent=2, ensure_ascii=False)

logger.info("=" * 60)
logger.info(f"✅ Stage 2 hoàn thành! Best Val F1-Macro: {best_f1_macro:.4f}")
logger.info(f"Per-label F1 (best epoch): {dict(zip(HARM_COLS, best_ckpt['per_label_f1']))}")
logger.info(f"Best thresholds: {threshold_dict}")
logger.info("📦 NEXT STEPS:")
logger.info("  1. Evaluate trên Test set bằng best_model.pt + best_thresholds.json")
logger.info("  2. Visualize train_history.csv")
logger.info("=" * 60)
