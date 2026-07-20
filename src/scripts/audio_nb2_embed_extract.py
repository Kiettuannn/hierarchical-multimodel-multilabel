"""
=============================================================================
NB-2: CLAP Embedding Extraction — Pre-cache Audio Embeddings
=============================================================================
Input (từ NB-1 Kaggle Dataset output):
  /kaggle/input/tiktok-audio-preprocessed/no_vocals_index.csv
  /kaggle/input/tiktok-audio-preprocessed/separated/htdemucs/{stem}/no_vocals.wav

Output (save thành Kaggle Dataset cho NB-3):
  /kaggle/working/audio_embeddings/
  ├── embeddings/
  │   └── {stem_name}.npy            ← vector 512-dim float32
  ├── embedding_index.csv            ← filename | stem_name | emb_path
  └── extraction_report.json

Model: laion/larger_clap_general (512-dim embedding, ~3GB VRAM)
Kaggle Setup:
  - Accelerator: GPU T4 (~3GB VRAM cho CLAP encoder)
  - Internet: ON (download model lần đầu ~800MB)
  - Runtime: ~30–60 phút cho 6286 files
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install transformers librosa -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import librosa
import json
import time
from pathlib import Path
from tqdm.auto import tqdm
import logging

from transformers import ClapModel, ClapProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/nb2_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
INDEX_CSV_PATH = "/kaggle/input/tiktok-audio-preprocessed/no_vocals_index.csv"
CLAP_MODEL_ID  = "laion/larger_clap_general"   # 512-dim, tốt hơn clap-htsat-fused
BATCH_SIZE     = 16                             # Tăng nếu còn VRAM, giảm nếu OOM
SAMPLE_RATE    = 16000                          # Phải khớp với Demucs output
MAX_DURATION_S = 60.0                           # Clip tối đa 60s (TikTok max)

OUTPUT_ROOT  = Path("/kaggle/working/audio_embeddings")
EMB_DIR      = OUTPUT_ROOT / "embeddings"
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ── Cell 3: Load CLAP model (chỉ audio encoder) ───────────────────────────────
logger.info(f"Loading CLAP model: {CLAP_MODEL_ID}")
clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
clap_model     = ClapModel.from_pretrained(CLAP_MODEL_ID).to(DEVICE)
clap_model.eval()

# Freeze toàn bộ — chỉ dùng để extract embeddings, không train
for p in clap_model.parameters():
    p.requires_grad_(False)

logger.info("CLAP model loaded and frozen ✅")
logger.info(f"Audio embedding dim: {clap_model.config.projection_dim}")

# ── Cell 4: Load index CSV ────────────────────────────────────────────────────
index_df = pd.read_csv(INDEX_CSV_PATH)
logger.info(f"Index CSV: {len(index_df):,} entries")
logger.info(f"Columns: {index_df.columns.tolist()}")

# ── Cell 5: Helper functions ──────────────────────────────────────────────────
def load_audio(wav_path: str, max_duration: float = MAX_DURATION_S) -> np.ndarray | None:
    """
    Load wav file với librosa. Clip tối đa max_duration giây.
    Returns float32 numpy array shape (samples,) ở 16kHz.
    """
    try:
        audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True,
                                 duration=max_duration)
        if len(audio) < 1600:   # < 0.1s → skip (file rỗng hoặc hỏng)
            return None
        return audio.astype(np.float32)
    except Exception as e:
        logger.warning(f"Load audio failed [{wav_path}]: {e}")
        return None


@torch.no_grad()
def extract_embeddings_batch(audio_batch: list[np.ndarray]) -> np.ndarray:
    """
    Extract CLAP audio embeddings cho một batch của audio arrays.
    Returns: numpy array shape (B, 512)
    """
    inputs = clap_processor(
        audios=audio_batch,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Lấy audio embedding (projected 512-dim)
    audio_embed = clap_model.get_audio_features(**inputs)  # (B, 512)
    return audio_embed.cpu().numpy().astype(np.float32)


# ── Cell 6: Main extraction loop (batch processing) ───────────────────────────
results = {
    "success"     : [],
    "load_failed" : [],
    "emb_failed"  : [],
    "skipped"     : [],
}

rows = index_df.to_dict("records")
t_start = time.time()

# Process in batches
for batch_start in tqdm(range(0, len(rows), BATCH_SIZE), desc="Extracting CLAP Embeddings"):
    batch_rows = rows[batch_start : batch_start + BATCH_SIZE]

    batch_audios  = []
    batch_stems   = []
    batch_filenames = []

    for row in batch_rows:
        stem_name     = str(row["stem_name"])
        filename      = str(row["filename"])
        no_vocals_path = str(row["no_vocals_path"])

        # Skip nếu đã có embedding
        emb_path = EMB_DIR / f"{stem_name}.npy"
        if emb_path.exists():
            results["skipped"].append(filename)
            continue

        audio = load_audio(no_vocals_path)
        if audio is None:
            results["load_failed"].append(filename)
            continue

        batch_audios.append(audio)
        batch_stems.append(stem_name)
        batch_filenames.append(filename)

    if not batch_audios:
        continue

    try:
        embeddings = extract_embeddings_batch(batch_audios)  # (B, 512)
        for i, (stem_name, filename) in enumerate(zip(batch_stems, batch_filenames)):
            emb_path = EMB_DIR / f"{stem_name}.npy"
            np.save(str(emb_path), embeddings[i])   # shape (512,)
            results["success"].append(filename)
    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        for filename in batch_filenames:
            results["emb_failed"].append(filename)

elapsed = time.time() - t_start
logger.info(f"Total time: {elapsed/60:.1f} min")


# ── Cell 7: Summary ───────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info(f"✅ Success    : {len(results['success']):,}")
logger.info(f"⏭️  Skipped    : {len(results['skipped']):,} (already cached)")
logger.info(f"❌ Load fail  : {len(results['load_failed']):,}")
logger.info(f"❌ Embed fail : {len(results['emb_failed']):,}")

actual_embs = list(EMB_DIR.glob("*.npy"))
logger.info(f"📁 .npy files : {len(actual_embs):,}")

# Verify embedding shape
if actual_embs:
    sample_emb = np.load(str(actual_embs[0]))
    logger.info(f"Sample embedding shape: {sample_emb.shape} dtype={sample_emb.dtype}")

# Save report
report = {
    "model"       : CLAP_MODEL_ID,
    "success"     : len(results["success"]),
    "skipped"     : len(results["skipped"]),
    "load_failed" : results["load_failed"],
    "emb_failed"  : results["emb_failed"],
    "total_npy"   : len(actual_embs),
    "elapsed_min" : round(elapsed / 60, 2),
}
with open(OUTPUT_ROOT / "extraction_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)


# ── Cell 8: Tạo embedding_index.csv cho NB-3 ──────────────────────────────────
"""
Schema: filename | stem_name | emb_path
NB-3 sẽ join embedding_index.csv với label CSV để load embeddings khi training.
"""
emb_index_rows = []
for emb_npy in EMB_DIR.glob("*.npy"):
    stem_name = emb_npy.stem   # e.g. "00030e61-7543965137888103711"
    # Tìm lại filename gốc (có .mp4) từ index_df
    match = index_df[index_df["stem_name"] == stem_name]
    if len(match) == 1:
        filename = match.iloc[0]["filename"]
    else:
        filename = stem_name + ".mp4"  # fallback

    emb_index_rows.append({
        "filename" : filename,
        "stem_name": stem_name,
        "emb_path" : str(emb_npy),
    })

emb_index_df = pd.DataFrame(emb_index_rows)
emb_index_csv = OUTPUT_ROOT / "embedding_index.csv"
emb_index_df.to_csv(emb_index_csv, index=False)

logger.info(f"✅ Embedding index: {emb_index_csv} ({len(emb_index_df):,} entries)")
logger.info("")
logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save output → Kaggle Dataset 'tiktok-audio-embeddings'")
logger.info("  2. Attach to NB-3 (audio_branch_train.py)")
logger.info("  3. NB-3 sẽ join embedding_index.csv với split_assignment.csv")
logger.info("=" * 60)
