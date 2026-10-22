"""
=============================================================================
Data Split — Chia Train/Val/Test theo IterativeStratification
=============================================================================
Input:
  data/label-origin.csv.csv  ← filename | 7 harm label cols

Output:
  data/splits/
  ├── split_assignment.csv   ← filename | split ('train'/'val'/'test')
  ├── stage1_train.csv       ← filename | is_harmful
  ├── stage1_val.csv
  ├── stage1_test.csv
  ├── stage2_train.csv       ← filename | 7 harm label cols (harmful only)
  ├── stage2_val.csv
  ├── stage2_test.csv
  └── split_report.json      ← thống kê phân phối nhãn

Thuật toán: IterativeStratification (skmultilearn)
Tỷ lệ: Train 70% / Val 15% / Test 15%

Yêu cầu:
  pip install scikit-multilearn
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install scikit-multilearn -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json
from pathlib import Path
from skmultilearn.model_selection import IterativeStratification

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LABEL_CSV  = "../../data/label-origin.csv.csv"
OUTPUT_DIR = "../../../data"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # = 1 - TRAIN - VAL

HARM_COLS = [
    "information_harm",
    "sexual_harm",
    "psychological_harm",
    "hate_harassment_harm",
    "clickbait_harm",
    "addictive_harm",
    "physical_harm",
]
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cell 3: Load và chuẩn bị dữ liệu ─────────────────────────────────────────
df = pd.read_csv(LABEL_CSV, dtype={"filename": str})
df["filename"] = df["filename"].str.strip()
df[HARM_COLS]  = df[HARM_COLS].fillna(0).astype(int)

# Tạo cột normal_content: 1 nếu tất cả 7 harm = 0
df["normal_content"] = (df[HARM_COLS].sum(axis=1) == 0).astype(int)
df["is_harmful"]     = 1 - df["normal_content"]

print(f"Total videos      : {len(df):,}")
print(f"Harmful videos    : {df['is_harmful'].sum():,}")
print(f"Normal videos     : {df['normal_content'].sum():,}")
print()
print("Phân phối từng nhãn:")
for col in HARM_COLS:
    n = df[col].sum()
    print(f"  {col:<30}: {n:,} ({n/len(df)*100:.1f}%)")

# ── Cell 4: IterativeStratification Split ────────────────────────────────────
# Ma trận stratification gồm 8 cột: 7 harm + 1 normal
STRAT_COLS = HARM_COLS + ["normal_content"]

X = np.arange(len(df)).reshape(-1, 1)  # index
y = df[STRAT_COLS].values               # (N, 8)

# Split 1: Tách Train (70%) vs Temp (30%)
# sample_distribution_per_fold=[0.3, 0.7]: fold_0=30%(temp), fold_1=70%(train)
# next() trả về (fold_0_idx, fold_1_idx) → unpack đúng thứ tự
splitter_1 = IterativeStratification(
    n_splits=2,
    order=2,
    sample_distribution_per_fold=[1 - TRAIN_RATIO, TRAIN_RATIO]
)
train_idx, temp_idx = next(splitter_1.split(X, y))  # FIX: đảo lại so với trước

X_temp = X[temp_idx]
y_temp = y[temp_idx]

# Split 2: Tách Val (50% of Temp = 15%) vs Test (50% of Temp = 15%)
splitter_2 = IterativeStratification(
    n_splits=2,
    order=2,
    sample_distribution_per_fold=[0.5, 0.5]
)
val_idx_local, test_idx_local = next(splitter_2.split(X_temp, y_temp))  # FIX: đảo lại

# Map local index về global index
val_idx  = temp_idx[val_idx_local]
test_idx = temp_idx[test_idx_local]

print(f"\nSplit sizes:")
print(f"  Train : {len(train_idx):,} ({len(train_idx)/len(df)*100:.1f}%)")
print(f"  Val   : {len(val_idx):,} ({len(val_idx)/len(df)*100:.1f}%)")
print(f"  Test  : {len(test_idx):,} ({len(test_idx)/len(df)*100:.1f}%)")

# ── Cell 5: Tạo split_assignment.csv ─────────────────────────────────────────
split_col = np.empty(len(df), dtype=object)
split_col[train_idx] = "train"
split_col[val_idx]   = "val"
split_col[test_idx]  = "test"

df["split"] = split_col

split_assignment = df[["filename", "split"]].copy()
split_assignment.to_csv(OUTPUT_DIR / "split_assignment.csv", index=False)
print(f"\nSaved: split_assignment.csv")

# ── Cell 6: Tạo Stage 1 CSVs ─────────────────────────────────────────────────
for split_name in ["train", "val", "test"]:
    mask    = df["split"] == split_name
    stage1  = df[mask][["filename", "is_harmful"]].copy()
    out_path = OUTPUT_DIR / f"stage1_{split_name}.csv"
    stage1.to_csv(out_path, index=False)
    n_harm   = stage1["is_harmful"].sum()
    n_normal = len(stage1) - n_harm
    print(f"stage1_{split_name}.csv : {len(stage1):,} rows | harmful={n_harm} | normal={n_normal}")

# ── Cell 7: Tạo Stage 2 CSVs (chỉ harmful samples) ───────────────────────────
print()
for split_name in ["train", "val", "test"]:
    mask   = (df["split"] == split_name) & (df["is_harmful"] == 1)
    stage2 = df[mask][["filename"] + HARM_COLS].copy()
    out_path = OUTPUT_DIR / f"stage2_{split_name}.csv"
    stage2.to_csv(out_path, index=False)
    print(f"stage2_{split_name}.csv : {len(stage2):,} harmful rows")
    for col in HARM_COLS:
        print(f"    {col:<30}: {stage2[col].sum()}")

# ── Cell 8: Save report ───────────────────────────────────────────────────────
report = {
    "total": len(df),
    "train": int(len(train_idx)),
    "val"  : int(len(val_idx)),
    "test" : int(len(test_idx)),
    "label_distribution": {
        col: int(df[col].sum()) for col in HARM_COLS + ["normal_content"]
    },
    "label_distribution_train": {
        col: int(df[df["split"] == "train"][col].sum()) for col in HARM_COLS
    },
    "label_distribution_val": {
        col: int(df[df["split"] == "val"][col].sum()) for col in HARM_COLS
    },
    "label_distribution_test": {
        col: int(df[df["split"] == "test"][col].sum()) for col in HARM_COLS
    },
}
with open(OUTPUT_DIR / "split_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n✅ Tất cả file đã được lưu vào: {OUTPUT_DIR.resolve()}")
print("📦 NEXT STEPS:")
print("  1. Upload data/splits/ lên Kaggle Dataset 'tiktok-data-splits'")
print("  2. Attach tới fusion_nb1_train.py")
