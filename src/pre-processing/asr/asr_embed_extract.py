"""
=============================================================================
NB-ASR-4: Text Embedding Extraction — PhoBERT từ ASR Transcript
=============================================================================
Input:
  /kaggle/input/tiktok-asr-processed/aux_asr.csv
    ← filename | cleaned_transcript | 7 harm signal cols

Output (save thành Kaggle Dataset cho Fusion NB):
  /kaggle/working/asr_embeddings/
  ├── embeddings/
  │   └── {stem_name}.npy        ← [768-d] float32 (PhoBERT [CLS] token)
  ├── asr_embed_index.csv         ← filename | stem_name | emb_path | is_zero_vec
  ├── asr_harm_priors.csv         ← filename | 7 harm signal cols (tabular features)
  └── asr_nb4_run.log

Model: vinai/phobert-base-v2 (768-dim)
Kaggle Setup:
  - Accelerator: GPU T4 x1
  - Internet: ON
  - Runtime: ~10-15 phút cho 6286 samples

Ghi chú:
  - stem_name = filename với .mp4 bị strip (key join với các modality khác)
  - Nếu cleaned_transcript rỗng/NaN → zero vector [768-d]
  - Max token length: 512 (transcript có thể dài)
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
        logging.FileHandler("/kaggle/working/asr_nb4_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_CSV     = "/kaggle/input/tiktok-asr-processed/aux_asr.csv"
PHOBERT_MODEL = "vinai/phobert-base-v2"
TEXT_COL      = "cleaned_transcript"
MAX_LENGTH    = 512   # Transcript ASR có thể dài hơn scene description
BATCH_SIZE    = 32    # Nhỏ hơn vì sequence dài hơn

HARM_SIGNAL_COLS = [
    "information_harm_signal",
    "sexual_harm_signal",
    "psychological_harm_signal",
    "hate_harassment_harm_signal",
    "clickbait_harm_signal",
    "addictive_harm_signal",
    "physical_harm_signal",
]

OUTPUT_ROOT = Path("/kaggle/working/asr_embeddings")
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
df["filename"]  = df["filename"].str.strip()
# stem_name: bỏ đuôi .mp4 để làm key join chung
df["stem_name"] = df["filename"].str.replace(r"\.mp4$", "", regex=True)
df[TEXT_COL]    = df[TEXT_COL].fillna("").str.strip()

logger.info(f"Total rows: {len(df):,}")
logger.info(f"Empty transcripts: {(df[TEXT_COL] == '').sum():,}")

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

for batch_start in tqdm(range(0, len(rows), BATCH_SIZE), desc="ASR PhoBERT Embed"):
    batch_rows = rows[batch_start: batch_start + BATCH_SIZE]

    batch_texts     = []
    batch_stems     = []
    batch_filenames = []
    batch_is_empty  = []

    for row in batch_rows:
        stem_name = str(row["stem_name"])
        filename  = str(row["filename"])
        emb_path  = EMB_DIR / f"{stem_name}.npy"

        if emb_path.exists():
            results["skipped"].append(stem_name)
            index_rows.append({"filename": filename, "stem_name": stem_name,
                                "emb_path": str(emb_path), "is_zero_vec": False})
            continue

        text     = str(row[TEXT_COL])
        is_empty = text == "" or text.lower() == "nan"

        batch_texts.append(text if not is_empty else "không có nội dung")
        batch_stems.append(stem_name)
        batch_filenames.append(filename)
        batch_is_empty.append(is_empty)

    if not batch_texts:
        continue

    embeddings = embed_texts(batch_texts)  # (B, 768)

    for i, (stem_name, filename, is_empty) in enumerate(
            zip(batch_stems, batch_filenames, batch_is_empty)):
        emb_path = EMB_DIR / f"{stem_name}.npy"
        emb      = embeddings[i]

        if is_empty:
            emb = np.zeros(768, dtype=np.float32)
            results["zero_vec"].append(stem_name)
        else:
            results["success"].append(stem_name)

        np.save(str(emb_path), emb)
        index_rows.append({"filename": filename, "stem_name": stem_name,
                            "emb_path": str(emb_path), "is_zero_vec": is_empty})

# ── Cell 7: Save harm priors (tabular features) ───────────────────────────────
# Lưu riêng 7 harm signal để dùng làm tabular feature trong Fusion Model
priors_df = df[["filename", "stem_name"] + HARM_SIGNAL_COLS].copy()
priors_df[HARM_SIGNAL_COLS] = priors_df[HARM_SIGNAL_COLS].fillna(0).astype(int)
priors_df.to_csv(OUTPUT_ROOT / "asr_harm_priors.csv", index=False)
logger.info(f"Harm priors saved: {len(priors_df):,} rows")

# ── Cell 8: Save index & report ───────────────────────────────────────────────
index_df = pd.DataFrame(index_rows)
index_df.to_csv(OUTPUT_ROOT / "asr_embed_index.csv", index=False)

actual_npy = list(EMB_DIR.glob("*.npy"))
logger.info("=" * 60)
logger.info(f"✅ Success    : {len(results['success']):,}")
logger.info(f"⬜ Zero vector: {len(results['zero_vec']):,} (empty transcript)")
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
logger.info("  1. Save output → Kaggle Dataset 'asr-embeddings'")
logger.info("  2. Attach tới fusion_nb1_train.py")
logger.info("=" * 60)
