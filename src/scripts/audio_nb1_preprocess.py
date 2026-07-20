"""
=============================================================================
NB-1: Audio Preprocessing — FFmpeg Extract + Demucs Source Separation
=============================================================================
Cấu trúc dữ liệu đầu vào (Kaggle):
  - Label CSV   : /kaggle/input/.../label-origin.csv.csv
                  Cột `filename` chứa tên file đầy đủ, ví dụ:
                    * "20260402174909.mp4"              (timestamp format)
                    * "00030e61-7543965137888103711.mp4" (hash-tiktokid format)
                    * "aa37d697-...-seg4.mp4"            (segment format)
  - Video dirs  : /kaggle/input/.../video_parts/part_01/ ... part_13/
                  File .mp4 nằm phẳng trong từng part, tên khớp cột `filename`

Output structure (save thành Kaggle Dataset cho NB-2):
  /kaggle/working/audio_preprocessed/
  ├── raw/
  │   └── {filename_no_ext}.wav          ← audio thô 16kHz mono
  └── separated/
      └── htdemucs/
          └── {filename_no_ext}/
              ├── vocals.wav             ← giọng nói (không dùng)
              └── no_vocals.wav          ← ÂM THANH NỀN ✅ → NB-2 input
  └── no_vocals_index.csv               ← index để NB-2 đọc
  └── preprocessing_report.json

Kaggle Setup:
  - Accelerator : GPU T4 (bắt buộc để Demucs chạy nhanh)
  - Internet    : ON (download Demucs model ~170MB lần đầu)
  - RAM         : ~16GB (đủ)
  - Disk        : ~20–40GB (6000 video × ~1–5MB mỗi audio)
  - Runtime ước tính: ~2–3 giờ cho 6286 videos
=============================================================================
"""

# ── Cell 1: Install & verify ──────────────────────────────────────────────────
# !pip install demucs -q
# !python -m demucs --help > /dev/null && echo "Demucs OK"
# !ffmpeg -version | head -n 1

# ── Cell 2: Imports & Config ───────────────────────────────────────────────────
import os
import subprocess
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import logging
import json
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/nb1_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — Chỉ cần thay đổi 2 giá trị này cho đúng với Kaggle Dataset của bạn
# ─────────────────────────────────────────────────────────────────────────────
LABEL_CSV_PATH  = "/kaggle/input/YOUR_DATASET_NAME/label-origin.csv.csv"
VIDEO_PARTS_DIR = "/kaggle/input/YOUR_DATASET_NAME/video_parts"
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_ROOT   = Path("/kaggle/working/audio_preprocessed")
RAW_DIR       = OUTPUT_ROOT / "raw"
SEPARATED_DIR = OUTPUT_ROOT / "separated"

SAMPLE_RATE   = 16000        # Hz — chuẩn CLAP / Demucs
DEMUCS_MODEL  = "htdemucs"  # Tốt nhất, download ~170MB lần đầu


# ── Cell 3: Setup directories ─────────────────────────────────────────────────
for d in [RAW_DIR, SEPARATED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
logger.info(f"Output root: {OUTPUT_ROOT}")

# ── Cell 4: Build video lookup table (filename → full_path) ───────────────────
"""
Scan tất cả part_XX/ để tạo dict {filename: full_path}.
Cấu trúc: video_parts/part_01/xxx.mp4, part_02/xxx.mp4, ...
filename trong label CSV khớp với basename của file video.
"""
logger.info("Building video lookup table from all parts...")
video_lookup: dict[str, Path] = {}

video_parts_root = Path(VIDEO_PARTS_DIR)
for part_dir in sorted(video_parts_root.iterdir()):
    if not part_dir.is_dir():
        continue
    for video_file in part_dir.glob("*.mp4"):
        fname = video_file.name  # e.g. "00030e61-7543965137888103711.mp4"
        if fname in video_lookup:
            logger.warning(f"Duplicate filename found: {fname} in {part_dir.name} (already in {video_lookup[fname].parent.name})")
        video_lookup[fname] = video_file

logger.info(f"Found {len(video_lookup):,} video files across all parts")

# ── Cell 5: Load label CSV ─────────────────────────────────────────────────────
df = pd.read_csv(LABEL_CSV_PATH)
logger.info(f"Label CSV: {len(df):,} rows | Columns: {df.columns.tolist()}")

# Validate filename format
sample = df["filename"].head(5).tolist()
logger.info(f"Sample filenames from CSV: {sample}")

# ── Cell 6: Cross-check CSV vs video files ─────────────────────────────────────
csv_filenames  = set(df["filename"].tolist())
disk_filenames = set(video_lookup.keys())

matched     = csv_filenames & disk_filenames
unmatched   = csv_filenames - disk_filenames
extra_disk  = disk_filenames - csv_filenames

logger.info(f"CSV entries        : {len(csv_filenames):,}")
logger.info(f"Disk video files   : {len(disk_filenames):,}")
logger.info(f"✅ Matched          : {len(matched):,}")
logger.info(f"❌ In CSV, not disk : {len(unmatched):,}  ← these will be skipped")
logger.info(f"ℹ️  On disk, no label: {len(extra_disk):,} ← ignored")

if unmatched:
    logger.warning(f"Sample unmatched: {list(unmatched)[:5]}")

# Chỉ process các file có cả label lẫn video
df_process = df[df["filename"].isin(matched)].copy().reset_index(drop=True)
logger.info(f"Will process {len(df_process):,} videos")

# ── Cell 7: FFmpeg audio extraction ───────────────────────────────────────────
def extract_audio_ffmpeg(video_path: Path, out_wav: Path) -> bool:
    """
    Trích xuất audio từ video → WAV 16kHz mono (chuẩn CLAP & Demucs).
    Idempotent: skip nếu file đã tồn tại.
    """
    if out_wav.exists() and out_wav.stat().st_size > 1000:
        return True  # Already done

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",                  # Mono
        "-ar", str(SAMPLE_RATE),     # 16000 Hz
        "-vn",                       # Drop video stream
        "-acodec", "pcm_s16le",      # 16-bit PCM WAV (lossless, Demucs-compatible)
        str(out_wav)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning(f"FFmpeg FAIL [{out_wav.stem}]: {result.stderr[-300:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(f"FFmpeg TIMEOUT [{out_wav.stem}]")
        return False
    except Exception as e:
        logger.error(f"FFmpeg ERROR [{out_wav.stem}]: {e}")
        return False


# ── Cell 8: Demucs source separation ──────────────────────────────────────────
def separate_audio_demucs(raw_wav: Path, stem_name: str) -> bool:
    """
    Tách âm thanh nền bằng Demucs htdemucs.
    - --two-stems=vocals : chỉ tách vocals / no_vocals (nhanh hơn 4-stem)
    - Output: SEPARATED_DIR/htdemucs/{stem_name}/no_vocals.wav

    stem_name = filename không có extension (vì Demucs dùng làm tên thư mục)
    """
    expected_output = SEPARATED_DIR / DEMUCS_MODEL / stem_name / "no_vocals.wav"
    if expected_output.exists() and expected_output.stat().st_size > 1000:
        return True  # Already done

    cmd = [
        "python", "-m", "demucs",
        "--two-stems=vocals",
        "-n", DEMUCS_MODEL,
        "--out", str(SEPARATED_DIR),
        str(raw_wav)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.warning(f"Demucs FAIL [{stem_name}]: {result.stderr[-400:]}")
            return False
        return expected_output.exists()
    except subprocess.TimeoutExpired:
        logger.warning(f"Demucs TIMEOUT [{stem_name}]")
        return False
    except Exception as e:
        logger.error(f"Demucs ERROR [{stem_name}]: {e}")
        return False


# ── Cell 9: Main processing loop ──────────────────────────────────────────────
results = {
    "success"       : [],
    "ffmpeg_failed" : [],
    "demucs_failed" : [],
}

t_start = time.time()

for _, row in tqdm(df_process.iterrows(), total=len(df_process), desc="Audio Preprocessing"):
    filename  = str(row["filename"])          # e.g. "00030e61-7543965137888103711.mp4"
    stem_name = Path(filename).stem           # e.g. "00030e61-7543965137888103711"
    video_path = video_lookup[filename]

    raw_wav = RAW_DIR / f"{stem_name}.wav"

    # Step 1: Extract raw audio via FFmpeg
    if not extract_audio_ffmpeg(video_path, raw_wav):
        results["ffmpeg_failed"].append(filename)
        continue

    # Step 2: Separate background music via Demucs
    if not separate_audio_demucs(raw_wav, stem_name):
        results["demucs_failed"].append(filename)
        continue

    results["success"].append(filename)

elapsed = time.time() - t_start
logger.info(f"Total time: {elapsed/60:.1f} minutes ({elapsed/len(df_process):.2f}s/video)")


# ── Cell 10: Summary & Validation ─────────────────────────────────────────────
logger.info("=" * 60)
logger.info(f"✅ Success        : {len(results['success']):,}")
logger.info(f"❌ FFmpeg failed  : {len(results['ffmpeg_failed']):,}")
logger.info(f"❌ Demucs failed  : {len(results['demucs_failed']):,}")

actual_no_vocals = list(SEPARATED_DIR.glob(f"{DEMUCS_MODEL}/*/no_vocals.wav"))
logger.info(f"📁 no_vocals files: {len(actual_no_vocals):,}")

# Save JSON report
report = {
    "total_in_csv"   : len(df),
    "matched_on_disk": len(matched),
    "processed"      : len(df_process),
    "success"        : len(results["success"]),
    "ffmpeg_failed"  : results["ffmpeg_failed"],
    "demucs_failed"  : results["demucs_failed"],
    "no_vocals_count": len(actual_no_vocals),
    "elapsed_min"    : round(elapsed / 60, 2),
}
with open(OUTPUT_ROOT / "preprocessing_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)


# ── Cell 11: Tạo no_vocals_index.csv cho NB-2 ─────────────────────────────────
"""
Schema: filename | stem_name | no_vocals_path
- filename   : tên gốc trong label CSV (e.g. "00030e61-xxx.mp4")
- stem_name  : tên không extension (e.g. "00030e61-xxx") — dùng để join với label
- no_vocals_path: đường dẫn đầy đủ đến no_vocals.wav
"""
index_rows = []
for filename in results["success"]:
    stem_name    = Path(filename).stem
    no_vocals_pth = SEPARATED_DIR / DEMUCS_MODEL / stem_name / "no_vocals.wav"
    if no_vocals_pth.exists():
        index_rows.append({
            "filename"      : filename,
            "stem_name"     : stem_name,
            "no_vocals_path": str(no_vocals_pth),
        })

index_df = pd.DataFrame(index_rows)
index_csv = OUTPUT_ROOT / "no_vocals_index.csv"
index_df.to_csv(index_csv, index=False)
logger.info(f"✅ Index CSV saved : {index_csv} ({len(index_df):,} entries)")
logger.info("")
logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Verify output: ls /kaggle/working/audio_preprocessed/")
logger.info("  2. Save notebook output → Kaggle Dataset (e.g. 'tiktok-audio-preprocessed')")
logger.info("  3. Attach that Dataset to NB-2 (audio_embed_extract.py)")
logger.info("=" * 60)
