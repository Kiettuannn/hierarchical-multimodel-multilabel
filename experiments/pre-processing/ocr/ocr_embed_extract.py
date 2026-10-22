"""
=============================================================================
NB-OCR-2: Text Embedding Extraction — PhoBERT từ OCR Text
=============================================================================
Input:
  /kaggle/input/tiktok-ocr-processed/aux_ocr.csv
    ← filename | cleaned_text | aux_psychological | aux_hate | aux_sexual |
      aux_addictive | aux_clickbait

Output (save thành Kaggle Dataset cho Fusion NB):
  /kaggle/working/ocr_embeddings/
  ├── embeddings/
  │   └── {filename}.npy         ← [768-d] float32 (PhoBERT [CLS] token)
  ├── ocr_embed_index.csv         ← filename | emb_path | is_zero_vec
  ├── ocr_aux_priors.csv          ← filename | 5 aux cols (tabular features)
  └── ocr_nb2_run.log

Model: vinai/phobert-base-v2 (768-dim)
Kaggle Setup:
  - Accelerator: GPU T4 x1
  - Internet: ON
  - Runtime: ~10-15 phút cho 6286 samples

Ghi chú:
  - filename trong aux_ocr.csv KHÔNG có .mp4 → dùng trực tiếp làm key
  - Nếu cleaned_text rỗng/NaN → zero vector [768-d]
  - Max token length: 256 (OCR text thường ngắn hơn ASR)
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install transformers -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import json
import logging
from pathlib import Path
from tqdm.auto import tqdm

from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/ocr_nb2_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_CSV     = "/kaggle/input/tiktok-ocr-processed/aux_ocr.csv"
PHOBERT_MODEL = "vinai/phobert-base-v2"
TEXT_COL      = "cleaned_text"
MAX_LENGTH    = 256
BATCH_SIZE    = 64

AUX_COLS = [
    "aux_psychological",
    "aux_hate",
    "aux_sexual",
    "aux_addictive",
    "aux_clickbait",
]

OUTPUT_ROOT = Path("/kaggle/working/ocr_embeddings")
EMB_DIR     = OUTPUT_ROOT / "embeddings"
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ── Cell 3: Load PhoBERT ──────────────────────────────────────────────────────
logger.info(f"Loading PhoBERT: {PHOBERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
model     = AutoModel.from_pretrained(PHOBERT_MODEL).to(DEVICE)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
logger.info("PhoBERT loaded and frozen ✅")

# ── Cell 4: Load CSV ──────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV, dtype={"filename": str})
df["filename"] = df["filename"].str.strip()
df[TEXT_COL]   = df[TEXT_COL].fillna("").str.strip()

logger.info(f"Total rows: {len(df):,}")
logger.info(f"Empty texts: {(df[TEXT_COL] == '').sum():,}")

# ── Cell 5: Embedding function ────────────────────────────────────────────────
@torch.no_grad()
def embed_texts(texts: list) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    cls_emb = outputs.last_hidden_state[:, 0, :]
    return cls_emb.cpu().numpy().astype(np.float32)

# ── Cell 6: Main extraction loop ─────────────────────────────────────────────
results    = {"success": [], "zero_vec": [], "skipped": []}
index_rows = []

rows = df.to_dict("records")

for batch_start in tqdm(range(0, len(rows), BATCH_SIZE), desc="OCR PhoBERT Embed"):
    batch_rows = rows[batch_start: batch_start + BATCH_SIZE]

    batch_texts     = []
    batch_filenames = []
    batch_is_empty  = []

    for row in batch_rows:
        filename = str(row["filename"])
        emb_path = EMB_DIR / f"{filename}.npy"

        if emb_path.exists():
            results["skipped"].append(filename)
            index_rows.append({"filename": filename,
                                "emb_path": str(emb_path), "is_zero_vec": False})
            continue

        text     = str(row[TEXT_COL])
        is_empty = text == "" or text.lower() == "nan"

        batch_texts.append(text if not is_empty else "không có văn bản")
        batch_filenames.append(filename)
        batch_is_empty.append(is_empty)

    if not batch_texts:
        continue

    embeddings = embed_texts(batch_texts)  # (B, 768)

    for i, (filename, is_empty) in enumerate(zip(batch_filenames, batch_is_empty)):
        emb_path = EMB_DIR / f"{filename}.npy"
        emb      = embeddings[i]

        if is_empty:
            emb = np.zeros(768, dtype=np.float32)
            results["zero_vec"].append(filename)
        else:
            results["success"].append(filename)

        np.save(str(emb_path), emb)
        index_rows.append({"filename": filename,
                            "emb_path": str(emb_path), "is_zero_vec": is_empty})

# ── Cell 7: Save aux priors (tabular features) ────────────────────────────────
# 5 aux signal dùng làm tabular feature trong Fusion Model
priors_df = df[["filename"] + AUX_COLS].copy()
priors_df[AUX_COLS] = priors_df[AUX_COLS].fillna(0).astype(int)
priors_df.to_csv(OUTPUT_ROOT / "ocr_aux_priors.csv", index=False)
logger.info(f"Aux priors saved: {len(priors_df):,} rows")

# ── Cell 8: Save index & report ───────────────────────────────────────────────
index_df = pd.DataFrame(index_rows)
index_df.to_csv(OUTPUT_ROOT / "ocr_embed_index.csv", index=False)

actual_npy = list(EMB_DIR.glob("*.npy"))
logger.info("=" * 60)
logger.info(f"✅ Success    : {len(results['success']):,}")
logger.info(f"⬜ Zero vector: {len(results['zero_vec']):,} (empty text)")
logger.info(f"⏭️  Skipped    : {len(results['skipped']):,} (already cached)")
logger.info(f"📁 .npy files : {len(actual_npy):,}")

if actual_npy:
    sample = np.load(str(actual_npy[0]))
    logger.info(f"Sample shape : {sample.shape} dtype={sample.dtype}")

report = {
    "model": PHOBERT_MODEL, "text_col": TEXT_COL,
    "success": len(results["success"]), "zero_vec": len(results["zero_vec"]),
    "skipped": len(results["skipped"]), "total_npy": len(actual_npy),
}
with open(OUTPUT_ROOT / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save output → Kaggle Dataset 'ocr-embeddings'")
logger.info("  2. Attach tới fusion_nb1_train.py")
logger.info("=" * 60)
