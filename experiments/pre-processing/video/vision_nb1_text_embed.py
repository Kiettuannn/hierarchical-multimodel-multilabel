"""
=============================================================================
NB-VISION-1: Text Embedding Extraction — PhoBERT từ Scene Description
=============================================================================
Input:
  /kaggle/input/tiktok-vision-scene-desc/scene_description.csv
    ← filename | scene_description

Output (save thành Kaggle Dataset cho NB-VISION-3):
  /kaggle/working/vision_text_emb/
  ├── embeddings/
  │   └── {filename}.npy         ← [768-d] float32 (PhoBERT [CLS] token)
  ├── text_embed_index.csv        ← filename | emb_path | is_zero_vec
  └── vision_nb1_run.log

Model: vinai/phobert-base-v2 (768-dim, ~0.5GB VRAM)
Kaggle Setup:
  - Accelerator: GPU T4 x1
  - Internet: ON (download model lần đầu ~500MB)
  - Runtime: ~10-15 phút cho 6286 samples

Ghi chú:
  - Nếu scene_description rỗng/NaN → lưu zero vector, đánh dấu is_zero_vec=True
  - Max token length: 256 (đủ cho 1-2 câu tiếng Việt)
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
        logging.FileHandler("/kaggle/working/vision_nb1_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INPUT_CSV      = "/kaggle/input/tiktok-vision-scene-desc/scene_description.csv"
PHOBERT_MODEL  = "vinai/phobert-base-v2"
MAX_LENGTH     = 256
BATCH_SIZE     = 64

OUTPUT_ROOT    = Path("/kaggle/working/vision_text_emb")
EMB_DIR        = OUTPUT_ROOT / "embeddings"
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ── Cell 3: Load PhoBERT ──────────────────────────────────────────────────────
logger.info(f"Loading PhoBERT model: {PHOBERT_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
model     = AutoModel.from_pretrained(PHOBERT_MODEL).to(DEVICE)
model.eval()

for p in model.parameters():
    p.requires_grad_(False)

logger.info("PhoBERT loaded and frozen ✅")
logger.info("Text embedding dim: 768")

# ── Cell 4: Load scene_description CSV ───────────────────────────────────────
df = pd.read_csv(INPUT_CSV, dtype={"filename": str})
df["filename"]          = df["filename"].str.strip()
df["scene_description"] = df["scene_description"].fillna("").str.strip()

logger.info(f"Total videos: {len(df):,}")
logger.info(f"Empty descriptions: {(df['scene_description'] == '').sum():,}")

# ── Cell 5: Embedding function ────────────────────────────────────────────────
@torch.no_grad()
def embed_texts(texts: list) -> np.ndarray:
    """
    Encode một batch text → [CLS] pooled output [B, 768].
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    # [CLS] token embedding: outputs.last_hidden_state[:, 0, :]
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.cpu().numpy().astype(np.float32)

# ── Cell 6: Main extraction loop ─────────────────────────────────────────────
results = {
    "success"  : [],
    "zero_vec" : [],
    "skipped"  : [],
}

rows       = df.to_dict("records")
index_rows = []

for batch_start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Extracting PhoBERT Embeddings"):
    batch_rows = rows[batch_start: batch_start + BATCH_SIZE]

    batch_texts     = []
    batch_filenames = []
    batch_is_empty  = []

    for row in batch_rows:
        filename = str(row["filename"])
        emb_path = EMB_DIR / f"{filename}.npy"

        # Skip nếu đã có embedding (resume-safe)
        if emb_path.exists():
            results["skipped"].append(filename)
            index_rows.append({
                "filename"   : filename,
                "emb_path"   : str(emb_path),
                "is_zero_vec": False,
            })
            continue

        text     = str(row["scene_description"])
        is_empty = (text == "") or (text.lower() == "nan")

        # Nếu rỗng dùng placeholder để tránh lỗi tokenizer
        batch_texts.append(text if not is_empty else "không có mô tả")
        batch_filenames.append(filename)
        batch_is_empty.append(is_empty)

    if not batch_texts:
        continue

    embeddings = embed_texts(batch_texts)  # (B, 768)

    for i, (filename, is_empty) in enumerate(zip(batch_filenames, batch_is_empty)):
        emb_path = EMB_DIR / f"{filename}.npy"
        emb      = embeddings[i]

        # Với text rỗng: override bằng zero vector để Fusion Model biết thiếu thông tin
        if is_empty:
            emb = np.zeros(768, dtype=np.float32)
            results["zero_vec"].append(filename)
        else:
            results["success"].append(filename)

        np.save(str(emb_path), emb)
        index_rows.append({
            "filename"   : filename,
            "emb_path"   : str(emb_path),
            "is_zero_vec": is_empty,
        })

# ── Cell 7: Save index & report ───────────────────────────────────────────────
index_df = pd.DataFrame(index_rows)
index_df.to_csv(OUTPUT_ROOT / "text_embed_index.csv", index=False)

actual_npy = list(EMB_DIR.glob("*.npy"))

logger.info("=" * 60)
logger.info(f"✅ Success    : {len(results['success']):,}")
logger.info(f"⬜ Zero vector: {len(results['zero_vec']):,} (empty description)")
logger.info(f"⏭️  Skipped    : {len(results['skipped']):,} (already cached)")
logger.info(f"📁 .npy files : {len(actual_npy):,}")

if actual_npy:
    sample = np.load(str(actual_npy[0]))
    logger.info(f"Sample shape : {sample.shape} dtype={sample.dtype}")

report = {
    "model"    : PHOBERT_MODEL,
    "success"  : len(results["success"]),
    "zero_vec" : len(results["zero_vec"]),
    "skipped"  : len(results["skipped"]),
    "total_npy": len(actual_npy),
}
with open(OUTPUT_ROOT / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save output → Kaggle Dataset 'vision-text-embeddings'")
logger.info("  2. Attach tới vision_nb3_combine.py")
logger.info("=" * 60)
