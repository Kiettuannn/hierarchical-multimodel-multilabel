"""
=============================================================================
NB-VISION-2: Visual Embedding Extraction — SigLIP2 từ Video Frames
=============================================================================
Input:
  /kaggle/input/datasets/kietkolat/hierarchical-multimodal-multilabel-frames/frames/
    └── {filename}/          ← thư mục chứa 8 frames của video
        ├── frame_001.jpg
        ├── frame_002.jpg
        └── ...

Output (save thành Kaggle Dataset cho NB-VISION-3):
  /kaggle/working/vision_image_emb/
  ├── embeddings/
  │   └── {filename}.npy     ← [768-d] float32 (SigLIP2 average pooled)
  ├── image_embed_index.csv   ← filename | emb_path | n_frames_used | is_zero_vec
  └── vision_nb2_run.log

Model: google/siglip2-base-patch16-224 (768-dim, ~3GB VRAM)
Kaggle Setup:
  - Accelerator: GPU T4 x1
  - Internet: ON (download model lần đầu ~1.1GB)
  - Runtime: ~30-45 phút cho 6286 videos

Ghi chú:
  - Lấy tối đa MAX_FRAMES_PER_VIDEO frames, phân bố đều theo tên file (sort)
  - Average Pool qua tất cả frames → 1 vector đại diện cho toàn video
  - Nếu folder không tồn tại hoặc không có ảnh → lưu zero vector
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install transformers Pillow -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import json
import logging
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

from transformers import AutoProcessor, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/vision_nb2_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FRAMES_ROOT       = Path("/kaggle/input/datasets/kietkolat/hierarchical-multimodal-multilabel-frames/frames")
SCENE_DESC_CSV    = "/kaggle/input/tiktok-vision-scene-desc/scene_description.csv"
SIGLIP_MODEL      = "google/siglip2-base-patch16-224"
MAX_FRAMES        = 8      # Số frames tối đa lấy mỗi video
BATCH_SIZE        = 8      # Số video xử lý song song (mỗi video có 8 frames → thực tế 64 ảnh/batch)

OUTPUT_ROOT       = Path("/kaggle/working/vision_image_emb")
EMB_DIR           = OUTPUT_ROOT / "embeddings"
# ─────────────────────────────────────────────────────────────────────────────

EMB_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE}")

# ── Cell 3: Load SigLIP2 ─────────────────────────────────────────────────────
logger.info(f"Loading SigLIP2: {SIGLIP_MODEL}")
processor  = AutoProcessor.from_pretrained(SIGLIP_MODEL)
siglip     = AutoModel.from_pretrained(SIGLIP_MODEL).to(DEVICE)
siglip.eval()

for p in siglip.parameters():
    p.requires_grad_(False)

logger.info("SigLIP2 loaded and frozen ✅")
logger.info("Image embedding dim: 768")

# ── Cell 4: Load filename list từ scene_description.csv ──────────────────────
df = pd.read_csv(SCENE_DESC_CSV, dtype={"filename": str})
df["filename"] = df["filename"].str.strip()
all_filenames  = df["filename"].tolist()

logger.info(f"Total videos to process: {len(all_filenames):,}")

# ── Cell 5: Helper functions ──────────────────────────────────────────────────
def load_frames(video_dir: Path, max_frames: int = MAX_FRAMES) -> list:
    """
    Load tối đa max_frames ảnh từ thư mục video.
    Lấy đều nhau theo thứ tự sort tên file.
    Returns: list of PIL.Image hoặc [] nếu không có ảnh.
    """
    img_paths = sorted(video_dir.glob("*.jpg")) + sorted(video_dir.glob("*.jpeg")) + sorted(video_dir.glob("*.png"))
    if not img_paths:
        return []

    # Lấy đều max_frames ảnh từ danh sách
    if len(img_paths) <= max_frames:
        selected = img_paths
    else:
        indices  = np.linspace(0, len(img_paths) - 1, max_frames, dtype=int)
        selected = [img_paths[i] for i in indices]

    images = []
    for p in selected:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
        except Exception as e:
            logger.warning(f"Cannot load image {p}: {e}")

    return images


@torch.no_grad()
def extract_video_embedding(images: list) -> np.ndarray:
    """
    Đẩy danh sách PIL Images qua SigLIP2 Vision Encoder.
    Average Pool qua tất cả frames → (768,).
    """
    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    vision_outputs = siglip.vision_model(**inputs)
    # pooler_output: (N_frames, 768)
    frame_embeddings = vision_outputs.pooler_output  # (N, 768)

    # Average Pool qua các frames
    video_embedding = frame_embeddings.mean(dim=0)   # (768,)
    return video_embedding.cpu().numpy().astype(np.float32)

# ── Cell 6: Main extraction loop ─────────────────────────────────────────────
results = {
    "success"    : [],
    "zero_vec"   : [],
    "skipped"    : [],
    "load_failed": [],
}

index_rows = []

for filename in tqdm(all_filenames, desc="Extracting SigLIP2 Embeddings"):
    emb_path = EMB_DIR / f"{filename}.npy"

    # Skip nếu đã có embedding (resume-safe)
    if emb_path.exists():
        results["skipped"].append(filename)
        index_rows.append({
            "filename"    : filename,
            "emb_path"    : str(emb_path),
            "n_frames_used": -1,
            "is_zero_vec" : False,
        })
        continue

    video_dir = FRAMES_ROOT / filename

    # Folder không tồn tại → zero vector
    if not video_dir.exists():
        logger.warning(f"Folder not found: {video_dir}")
        emb = np.zeros(768, dtype=np.float32)
        np.save(str(emb_path), emb)
        results["zero_vec"].append(filename)
        index_rows.append({
            "filename"    : filename,
            "emb_path"    : str(emb_path),
            "n_frames_used": 0,
            "is_zero_vec" : True,
        })
        continue

    images = load_frames(video_dir, MAX_FRAMES)

    # Không có ảnh nào → zero vector
    if not images:
        logger.warning(f"No images in folder: {video_dir}")
        emb = np.zeros(768, dtype=np.float32)
        np.save(str(emb_path), emb)
        results["zero_vec"].append(filename)
        index_rows.append({
            "filename"    : filename,
            "emb_path"    : str(emb_path),
            "n_frames_used": 0,
            "is_zero_vec" : True,
        })
        continue

    try:
        emb = extract_video_embedding(images)  # (768,)
        np.save(str(emb_path), emb)
        results["success"].append(filename)
        index_rows.append({
            "filename"    : filename,
            "emb_path"    : str(emb_path),
            "n_frames_used": len(images),
            "is_zero_vec" : False,
        })
    except Exception as e:
        logger.error(f"Embedding failed for {filename}: {e}")
        results["load_failed"].append(filename)

# ── Cell 7: Save index & report ───────────────────────────────────────────────
index_df = pd.DataFrame(index_rows)
index_df.to_csv(OUTPUT_ROOT / "image_embed_index.csv", index=False)

actual_npy = list(EMB_DIR.glob("*.npy"))

logger.info("=" * 60)
logger.info(f"✅ Success    : {len(results['success']):,}")
logger.info(f"⬜ Zero vector: {len(results['zero_vec']):,} (missing folder/frames)")
logger.info(f"⏭️  Skipped    : {len(results['skipped']):,} (already cached)")
logger.info(f"❌ Failed     : {len(results['load_failed']):,}")
logger.info(f"📁 .npy files : {len(actual_npy):,}")

if actual_npy:
    sample = np.load(str(actual_npy[0]))
    logger.info(f"Sample shape : {sample.shape} dtype={sample.dtype}")

report = {
    "model"      : SIGLIP_MODEL,
    "max_frames" : MAX_FRAMES,
    "success"    : len(results["success"]),
    "zero_vec"   : len(results["zero_vec"]),
    "skipped"    : len(results["skipped"]),
    "load_failed": results["load_failed"],
    "total_npy"  : len(actual_npy),
}
with open(OUTPUT_ROOT / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save output → Kaggle Dataset 'vision-image-embeddings'")
logger.info("  2. Attach tới vision_nb3_combine.py")
logger.info("=" * 60)
