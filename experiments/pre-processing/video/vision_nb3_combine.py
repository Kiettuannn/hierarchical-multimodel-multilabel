"""
=============================================================================
NB-VISION-3: Combine Vision Embeddings — SigLIP2 Image + PhoBERT Text
=============================================================================
Input:
  /kaggle/input/vision-image-embeddings/embeddings/{filename}.npy  ← [768-d]
  /kaggle/input/vision-text-embeddings/embeddings/{filename}.npy   ← [768-d]
  /kaggle/input/tiktok-vision-scene-desc/scene_description.csv     ← filename list

Output (save thành Kaggle Dataset cho Fusion NB):
  /kaggle/working/vision_embeddings_final/
  ├── embeddings/
  │   └── {filename}.npy        ← [1536-d] float32 (concat image + text)
  ├── vision_index.csv           ← filename | emb_path | has_image | has_text
  └── vision_nb3_run.log

Chiều của vector cuối: 1536-d = 768 (SigLIP2) + 768 (PhoBERT)
  [image_emb | text_emb]
   0 ....767   768...1535

Ghi chú:
  - Nếu thiếu image emb hoặc text emb → dùng zero vector cho phần bị thiếu
  - Vector cuối luôn có shape (1536,) nhất quán cho toàn bộ dataset
=============================================================================
"""

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/vision_nb3_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
IMAGE_EMB_DIR  = Path("/kaggle/input/vision-image-embeddings/embeddings")
TEXT_EMB_DIR   = Path("/kaggle/input/vision-text-embeddings/embeddings")
SCENE_DESC_CSV = "/kaggle/input/tiktok-vision-scene-desc/scene_description.csv"

OUTPUT_ROOT    = Path("/kaggle/working/vision_embeddings_final")
EMB_DIR        = OUTPUT_ROOT / "embeddings"

IMAGE_DIM      = 768
TEXT_DIM       = 768
FINAL_DIM      = IMAGE_DIM + TEXT_DIM  # 1536
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIR.mkdir(parents=True, exist_ok=True)

# ── Cell 2: Load filename list ────────────────────────────────────────────────
df = pd.read_csv(SCENE_DESC_CSV, dtype={"filename": str})
df["filename"] = df["filename"].str.strip()
all_filenames  = df["filename"].tolist()

logger.info(f"Total videos: {len(all_filenames):,}")
logger.info(f"Final embedding dim: {FINAL_DIM}")

# ── Cell 3: Main combine loop ─────────────────────────────────────────────────
results = {
    "both"        : [],
    "image_only"  : [],
    "text_only"   : [],
    "neither"     : [],
    "skipped"     : [],
}

index_rows = []

for filename in tqdm(all_filenames, desc="Combining Vision Embeddings"):
    out_path = EMB_DIR / f"{filename}.npy"

    # Skip nếu đã combine (resume-safe)
    if out_path.exists():
        results["skipped"].append(filename)
        index_rows.append({
            "filename" : filename,
            "emb_path" : str(out_path),
            "has_image": True,
            "has_text" : True,
        })
        continue

    image_path = IMAGE_EMB_DIR / f"{filename}.npy"
    text_path  = TEXT_EMB_DIR  / f"{filename}.npy"

    has_image = image_path.exists()
    has_text  = text_path.exists()

    # Load hoặc dùng zero vector nếu thiếu
    image_emb = np.load(str(image_path)) if has_image else np.zeros(IMAGE_DIM, dtype=np.float32)
    text_emb  = np.load(str(text_path))  if has_text  else np.zeros(TEXT_DIM,  dtype=np.float32)

    # Kiểm tra shape hợp lệ
    if image_emb.shape != (IMAGE_DIM,):
        logger.warning(f"Unexpected image emb shape {image_emb.shape} for {filename}, using zeros")
        image_emb = np.zeros(IMAGE_DIM, dtype=np.float32)
        has_image = False

    if text_emb.shape != (TEXT_DIM,):
        logger.warning(f"Unexpected text emb shape {text_emb.shape} for {filename}, using zeros")
        text_emb = np.zeros(TEXT_DIM, dtype=np.float32)
        has_text = False

    # Concatenate: [image_emb | text_emb] → (1536,)
    final_emb = np.concatenate([image_emb, text_emb])
    assert final_emb.shape == (FINAL_DIM,), f"Shape mismatch: {final_emb.shape}"

    np.save(str(out_path), final_emb)

    # Track kết quả
    if has_image and has_text:
        results["both"].append(filename)
    elif has_image:
        results["image_only"].append(filename)
    elif has_text:
        results["text_only"].append(filename)
    else:
        results["neither"].append(filename)

    index_rows.append({
        "filename" : filename,
        "emb_path" : str(out_path),
        "has_image": has_image,
        "has_text" : has_text,
    })

# ── Cell 4: Save index & report ───────────────────────────────────────────────
index_df = pd.DataFrame(index_rows)
index_df.to_csv(OUTPUT_ROOT / "vision_index.csv", index=False)

actual_npy = list(EMB_DIR.glob("*.npy"))

logger.info("=" * 60)
logger.info(f"✅ Both emb    : {len(results['both']):,}")
logger.info(f"🖼️  Image only  : {len(results['image_only']):,}")
logger.info(f"📝 Text only   : {len(results['text_only']):,}")
logger.info(f"⬜ Neither     : {len(results['neither']):,} (all zeros)")
logger.info(f"⏭️  Skipped     : {len(results['skipped']):,} (already cached)")
logger.info(f"📁 .npy files  : {len(actual_npy):,}")

if actual_npy:
    sample = np.load(str(actual_npy[0]))
    logger.info(f"Sample shape  : {sample.shape} dtype={sample.dtype}")
    logger.info(f"Has NaN       : {np.isnan(sample).any()}")

report = {
    "final_dim"  : FINAL_DIM,
    "both"       : len(results["both"]),
    "image_only" : len(results["image_only"]),
    "text_only"  : len(results["text_only"]),
    "neither"    : len(results["neither"]),
    "skipped"    : len(results["skipped"]),
    "total_npy"  : len(actual_npy),
}
with open(OUTPUT_ROOT / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save output → Kaggle Dataset 'vision-embeddings-final'")
logger.info("  2. Attach tới fusion_nb1_train.py")
logger.info(f"  3. Mỗi video có vision embedding shape: ({FINAL_DIM},)")
logger.info("     Cấu trúc: [SigLIP2 image 0:768 | PhoBERT text 768:1536]")
logger.info("=" * 60)
