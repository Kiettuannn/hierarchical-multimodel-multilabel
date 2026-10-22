"""
=============================================================================
FUSION NB-5: Evaluation — Stage 1 + Stage 2 + Cascade Inference
=============================================================================
Input:
  - stage1-checkpoints/best_model.pt
  - stage2-checkpoints/best_model.pt
  - stage2-checkpoints/best_thresholds.json
  - tiktok-data-splits/stage1_test.csv
  - tiktok-data-splits/stage2_test.csv
  - (+ tất cả embedding datasets)

Output:
  /kaggle/working/evaluation/
  ├── stage1_test_report.json     ← F1, ROC-AUC, Precision, Recall
  ├── stage1_confusion_matrix.csv
  ├── stage2_test_report.json     ← F1-Macro, per-label F1, Classification Report
  ├── cascade_test_report.json    ← Kết quả Cascade (Stage1 → Stage2) trên test set
  └── evaluation_run.log

Cascade logic:
  Video → Stage 1 → P_harmful < threshold → normal (tất cả nhãn = 0)
                  → P_harmful >= threshold → Stage 2 → 7 nhãn
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
import json
import logging
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

from fusion_nb1_dataloader import make_loader, HARM_COLS
from fusion_nb2_model import FusionMLP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/evaluation_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
STAGE1_CKPT      = "/kaggle/input/stage1-checkpoints/best_model.pt"
STAGE2_CKPT      = "/kaggle/input/stage2-checkpoints/best_model.pt"
STAGE2_THRESHOLD = "/kaggle/input/stage2-checkpoints/best_thresholds.json"

STAGE1_TEST_CSV  = "/kaggle/input/tiktok-data-splits/stage1_test.csv"
STAGE2_TEST_CSV  = "/kaggle/input/tiktok-data-splits/stage2_test.csv"
LABEL_CSV        = "/kaggle/input/tiktok-data-splits/split_assignment.csv"

# Ngưỡng Stage 1 phân loại harmful (có thể điều chỉnh)
STAGE1_THRESHOLD = 0.5

OUTPUT_DIR  = Path("/kaggle/working/evaluation")
BATCH_SIZE  = 256   # Lớn hơn train vì chỉ inference, không backward
DROPOUT     = 0.3
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ── Cell 3: Load thresholds Stage 2 ──────────────────────────────────────────
with open(STAGE2_THRESHOLD, "r") as f:
    stage2_thresholds = json.load(f)  # {label: threshold}
threshold_array = np.array([stage2_thresholds[col] for col in HARM_COLS])
logger.info(f"Stage 2 thresholds: {stage2_thresholds}")

# ── Cell 4: Load models ───────────────────────────────────────────────────────
# Stage 1 model
stage1_model = FusionMLP(out_size=1, dropout=DROPOUT).to(DEVICE)
ckpt1 = torch.load(STAGE1_CKPT, map_location=DEVICE)
stage1_model.load_state_dict(ckpt1["model_state"])
stage1_model.eval()
logger.info(f"Stage 1 loaded (Val ROC-AUC={ckpt1.get('val_roc_auc', '?')})")

# Stage 2 model
stage2_model = FusionMLP(out_size=7, dropout=DROPOUT).to(DEVICE)
ckpt2 = torch.load(STAGE2_CKPT, map_location=DEVICE)
stage2_model.load_state_dict(ckpt2["model_state"])
stage2_model.eval()
logger.info(f"Stage 2 loaded (Val F1-Macro={ckpt2.get('val_f1_macro', '?')})")

# ── Cell 5: Inference helper ──────────────────────────────────────────────────
@torch.no_grad()
def run_inference(loader, model, device=DEVICE):
    """
    Chạy inference và trả về (probs, labels).
    probs: (N, out_size) numpy array
    labels: (N, out_size) numpy array
    """
    all_probs  = []
    all_labels = []

    for batch in tqdm(loader, desc="Inference", leave=False):
        vision  = batch["vision"].to(device)
        asr     = batch["asr"].to(device)
        ocr     = batch["ocr"].to(device)
        clap    = batch["clap"].to(device)
        tabular = batch["tabular"].to(device)
        labels  = batch["label"]

        logits = model(vision, asr, ocr, clap, tabular)
        probs  = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Đánh giá Stage 1 trên test set
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("📊 SECTION 1: STAGE 1 TEST EVALUATION")
logger.info("=" * 60)

stage1_test_loader = make_loader(
    STAGE1_TEST_CSV, mode="stage1",
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

s1_probs, s1_labels = run_inference(stage1_test_loader, stage1_model)
s1_probs  = s1_probs.flatten()
s1_labels = s1_labels.flatten().astype(int)
s1_preds  = (s1_probs >= STAGE1_THRESHOLD).astype(int)

s1_f1        = f1_score(s1_labels, s1_preds, zero_division=0)
s1_precision = precision_score(s1_labels, s1_preds, zero_division=0)
s1_recall    = recall_score(s1_labels, s1_preds, zero_division=0)
s1_roc_auc   = roc_auc_score(s1_labels, s1_probs)
s1_cm        = confusion_matrix(s1_labels, s1_preds)

logger.info(f"F1-Binary   : {s1_f1:.4f}")
logger.info(f"Precision   : {s1_precision:.4f}")
logger.info(f"Recall      : {s1_recall:.4f}")
logger.info(f"ROC-AUC     : {s1_roc_auc:.4f}")
logger.info(f"Confusion Matrix:\n{s1_cm}")

# Tìm threshold tối ưu cho Stage 1 trên test set
best_t1, best_f1_t1 = STAGE1_THRESHOLD, s1_f1
for t in np.arange(0.1, 0.9, 0.05):
    preds = (s1_probs >= t).astype(int)
    f1    = f1_score(s1_labels, preds, zero_division=0)
    if f1 > best_f1_t1:
        best_f1_t1, best_t1 = f1, t
logger.info(f"Best threshold (test) : {best_t1:.2f} → F1={best_f1_t1:.4f}")

# Save Stage 1 report
s1_report = {
    "threshold"    : float(STAGE1_THRESHOLD),
    "best_threshold_on_test": float(round(best_t1, 2)),
    "f1_binary"    : float(s1_f1),
    "precision"    : float(s1_precision),
    "recall"       : float(s1_recall),
    "roc_auc"      : float(s1_roc_auc),
    "confusion_matrix": s1_cm.tolist(),
    "n_test"       : int(len(s1_labels)),
    "n_harmful"    : int(s1_labels.sum()),
    "n_normal"     : int(len(s1_labels) - s1_labels.sum()),
}
with open(OUTPUT_DIR / "stage1_test_report.json", "w") as f:
    json.dump(s1_report, f, indent=2)

pd.DataFrame(
    s1_cm,
    index=["Actual Normal", "Actual Harmful"],
    columns=["Pred Normal", "Pred Harmful"]
).to_csv(OUTPUT_DIR / "stage1_confusion_matrix.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Đánh giá Stage 2 độc lập (chỉ trên harmful videos)
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("📊 SECTION 2: STAGE 2 TEST EVALUATION (harmful only)")
logger.info("=" * 60)

stage2_test_loader = make_loader(
    STAGE2_TEST_CSV, mode="stage2",
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

s2_probs, s2_labels = run_inference(stage2_test_loader, stage2_model)
s2_labels = s2_labels.astype(int)
s2_preds  = (s2_probs >= threshold_array).astype(int)  # Per-label threshold

s2_f1_macro = f1_score(s2_labels, s2_preds, average="macro", zero_division=0)
s2_f1_micro = f1_score(s2_labels, s2_preds, average="micro", zero_division=0)
s2_per_label_f1 = f1_score(s2_labels, s2_preds, average=None, zero_division=0)

logger.info(f"F1-Macro    : {s2_f1_macro:.4f}")
logger.info(f"F1-Micro    : {s2_f1_micro:.4f}")
logger.info("Per-label F1:")
for col, f1 in zip(HARM_COLS, s2_per_label_f1):
    n_pos = s2_labels[:, HARM_COLS.index(col)].sum()
    logger.info(f"  {col:<30}: {f1:.4f}  (n_pos={n_pos})")

logger.info("\nClassification Report:")
logger.info(classification_report(
    s2_labels, s2_preds,
    target_names=[c.replace("_harm", "") for c in HARM_COLS],
    zero_division=0
))

# Save Stage 2 report
s2_report = {
    "thresholds_used" : stage2_thresholds,
    "f1_macro"        : float(s2_f1_macro),
    "f1_micro"        : float(s2_f1_micro),
    "per_label"       : {
        col: {
            "f1"       : float(f1),
            "precision": float(precision_score(s2_labels[:, i], s2_preds[:, i], zero_division=0)),
            "recall"   : float(recall_score(s2_labels[:, i], s2_preds[:, i], zero_division=0)),
            "n_pos_test": int(s2_labels[:, i].sum()),
        }
        for i, (col, f1) in enumerate(zip(HARM_COLS, s2_per_label_f1))
    },
    "n_test": int(len(s2_labels)),
}
with open(OUTPUT_DIR / "stage2_test_report.json", "w") as f:
    json.dump(s2_report, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Cascade Inference (Stage1 → Stage2) trên toàn bộ test set
# ═══════════════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("📊 SECTION 3: CASCADE INFERENCE (End-to-End)")
logger.info("=" * 60)

# Load toàn bộ test set (dùng stage1_test.csv làm gốc — có đủ mọi video)
cascade_loader = make_loader(
    STAGE1_TEST_CSV, mode="stage1",
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# Bước 1: Chạy Stage 1 trên toàn bộ test set
s1_probs_all, s1_labels_all = run_inference(cascade_loader, stage1_model)
s1_probs_all  = s1_probs_all.flatten()
s1_labels_all = s1_labels_all.flatten().astype(int)

harmful_mask = (s1_probs_all >= STAGE1_THRESHOLD)  # Boolean mask
n_predicted_harmful = harmful_mask.sum()
logger.info(f"Stage 1 predicted harmful: {n_predicted_harmful} / {len(s1_labels_all)}")

# Bước 2: Load stage2_test CSV để lấy ground truth 7 nhãn cho test set
# Join qua filename
stage1_df = pd.read_csv(STAGE1_TEST_CSV, dtype={"filename": str})
stage2_df = pd.read_csv(STAGE2_TEST_CSV, dtype={"filename": str})

# Merge để có ground truth 7 nhãn cho toàn bộ test set (normal = 0 cho tất cả nhãn)
full_test_df = stage1_df[["filename", "is_harmful"]].copy()
full_test_df = full_test_df.merge(
    stage2_df[["filename"] + HARM_COLS], on="filename", how="left"
)
full_test_df[HARM_COLS] = full_test_df[HARM_COLS].fillna(0).astype(int)
gt_labels = full_test_df[HARM_COLS].values  # (N, 7) ground truth

# Bước 3: Khởi tạo kết quả cascade = tất cả 0
cascade_preds = np.zeros((len(s1_labels_all), 7), dtype=int)

# Bước 4: Với các video Stage 1 predict là harmful → chạy Stage 2
# Cần tạo sub-loader cho chỉ những video đó
# Cách đơn giản: chạy Stage 2 trên stage2_test toàn bộ, rồi map lại
stage2_all_loader = make_loader(
    STAGE1_TEST_CSV, mode="stage2",
    batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

# Load Stage 2 model chạy trên toàn bộ test (kể cả normal)
# Chỉ apply kết quả vào những row mà Stage 1 predict harmful
with torch.no_grad():
    s2_probs_all = []
    for batch in tqdm(stage2_all_loader, desc="Stage 2 inference (all test)", leave=False):
        vision  = batch["vision"].to(DEVICE)
        asr     = batch["asr"].to(DEVICE)
        ocr     = batch["ocr"].to(DEVICE)
        clap    = batch["clap"].to(DEVICE)
        tabular = batch["tabular"].to(DEVICE)
        logits  = stage2_model(vision, asr, ocr, clap, tabular)
        probs   = torch.sigmoid(logits).cpu().numpy()
        s2_probs_all.append(probs)

s2_probs_all = np.vstack(s2_probs_all)  # (N, 7)

# Apply cascade: chỉ gán nhãn Stage 2 cho video được Stage 1 predict là harmful
cascade_preds[harmful_mask] = (
    s2_probs_all[harmful_mask] >= threshold_array
).astype(int)

# Đánh giá cascade
cascade_f1_macro = f1_score(gt_labels, cascade_preds, average="macro", zero_division=0)
cascade_f1_micro = f1_score(gt_labels, cascade_preds, average="micro", zero_division=0)
cascade_per_f1   = f1_score(gt_labels, cascade_preds, average=None, zero_division=0)

logger.info(f"Cascade F1-Macro : {cascade_f1_macro:.4f}")
logger.info(f"Cascade F1-Micro : {cascade_f1_micro:.4f}")
logger.info("Cascade Per-label F1:")
for col, f1 in zip(HARM_COLS, cascade_per_f1):
    logger.info(f"  {col:<30}: {f1:.4f}")

logger.info("\nCascade Classification Report:")
logger.info(classification_report(
    gt_labels, cascade_preds,
    target_names=[c.replace("_harm", "") for c in HARM_COLS],
    zero_division=0
))

# Save Cascade report
cascade_report = {
    "stage1_threshold" : float(STAGE1_THRESHOLD),
    "stage2_thresholds": stage2_thresholds,
    "n_test_total"     : int(len(s1_labels_all)),
    "n_predicted_harmful": int(n_predicted_harmful),
    "cascade_f1_macro" : float(cascade_f1_macro),
    "cascade_f1_micro" : float(cascade_f1_micro),
    "cascade_per_label": {
        col: float(f1) for col, f1 in zip(HARM_COLS, cascade_per_f1)
    },
}
with open(OUTPUT_DIR / "cascade_test_report.json", "w") as f:
    json.dump(cascade_report, f, indent=2, ensure_ascii=False)

# ── Cell 6: Summary ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("📋 EVALUATION SUMMARY")
logger.info("=" * 60)
logger.info(f"Stage 1  | F1-Binary={s1_f1:.4f} | ROC-AUC={s1_roc_auc:.4f}")
logger.info(f"Stage 2  | F1-Macro={s2_f1_macro:.4f} (harmful only)")
logger.info(f"Cascade  | F1-Macro={cascade_f1_macro:.4f} (full test set, end-to-end)")
logger.info("=" * 60)
logger.info(f"✅ Tất cả report đã lưu vào: {OUTPUT_DIR}")
