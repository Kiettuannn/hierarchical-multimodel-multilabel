"""
=============================================================================
NB-ASR-1: Transcript Extraction — ChunkFormer
=============================================================================
Input (từ Kaggle Dataset 'tiktok-vocal-audio'):
  /kaggle/input/tiktok-vocal-audio/{part}/vocal/{stem_name}/vocals.wav

Output (save thành Kaggle Dataset cho NB-ASR-2):
  /kaggle/working/asr_transcripts/
  ├── asr_transcripts.csv        ← main output
  └── asr_nb1_run.log

Schema asr_transcripts.csv:
  filename | raw_transcript | asr_confidence | has_uncertain | duration_s | status

Model: chunkformer

=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install chunkformer librosa -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import librosa
import json
import time
import logging
import warnings
from pathlib import Path
from tqdm.auto import tqdm

from chunkformer import ChunkFormerModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/asr_nb1_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ASR_MODEL_ID   = "khanhld/chunkformer-ctc-large-vie"
SAMPLE_RATE    = 16000          # PhoWhisper yêu cầu 16kHz
MAX_DURATION_S = 60.0           # Clip tối đa 60s (TikTok max length)

# ── PARALLEL SPLIT CONFIG ─────────────────────────────────────────────────────
# Chạy 2 notebooks song song để hoàn thành trong ~10-12h thay vì 22-25h:
#   Notebook A: PART = "A"  →  xử lý nửa đầu (file 0 → N//2)
#   Notebook B: PART = "B"  →  xử lý nửa sau (file N//2 → N)
# Sau khi cả 2 xong → merge 2 CSV thành 1 (Cell cuối script)
PART = "A"   # ← ĐỔI THÀNH "B" khi chạy notebook thứ 2
# ─────────────────────────────────────────────────────────────────────────────

# Uncertainty thresholds
UNCERTAINTY_LOGPROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD           = 0.5

OUTPUT_ROOT = Path("/kaggle/working/asr_transcripts")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV  = OUTPUT_ROOT / f"asr_transcripts_part{PART}.csv"   # part A hoặc B
SAVE_EVERY  = 100
# ─────────────────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
logger.info(f"Device: {DEVICE} | dtype: {TORCH_DTYPE}")

# ── Cell 3: Load ChunkFormer ──────────────────────────────────────────────────
logger.info(f"Loading ASR model: {ASR_MODEL_ID}")

model = ChunkFormerModel.from_pretrained(ASR_MODEL_ID)
model.to(DEVICE)
model.eval()

logger.info("ChunkFormer loaded ✅")

# ── Cell 4: Build file index ──────────────────────────────────────────────────
"""
Scan tất cả vocals.wav từ các Kaggle Dataset được attach.
Cấu trúc mong đợi:
  /kaggle/input/{dataset_name}/.../vocal/{stem_name}/vocals.wav
"""
logger.info("Scanning for vocals.wav files...")

base_input_dir = Path("/kaggle/input")
vocals_files   = list(base_input_dir.rglob("vocals.wav"))

index_rows = []
for vocals_path in vocals_files:
    stem_name = vocals_path.parent.name
    filename  = f"{stem_name}.mp4"
    index_rows.append({
        "filename"   : filename,
        "stem_name"  : stem_name,
        "vocals_path": str(vocals_path),
    })

index_df = pd.DataFrame(index_rows)
logger.info(f"Found {len(index_df):,} vocal files.")

# ── Cell 5: Helper functions ──────────────────────────────────────────────────
def get_audio_duration(wav_path: str) -> float:
    """
    Lấy độ dài file audio (giây) nhanh chóng bằng librosa.
    """
    try:
        return librosa.get_duration(path=wav_path)
    except Exception as e:
        logger.warning(f"Lỗi lấy duration [{wav_path}]: {e}")
        return 0.0


def calculate_confidence(transcript: str, duration_s: float) -> tuple[float, bool]:
    """
    Đánh giá độ tự tin (heuristic) dựa trên các lỗi phổ biến của CTC model.
    - Lặp từ liên tục (CTC Loop / stuttering do nhiễu).
    - Tốc độ sinh từ bất thường (quá dày đặc hoặc quá thưa thớt).
    
    Returns:
        (confidence_score: float, has_uncertain: bool)
    """
    if not transcript or duration_s <= 0.5:
        return 0.0, False
        
    words = transcript.split()
    if len(words) == 0:
        return 0.0, False
        
    confidence = 1.0
    has_uncertain = False
    
    # 1. Tốc độ nói (Words per second - WPS)
    # Người Việt nói trung bình 2-4 từ/giây. 
    wps = len(words) / duration_s
    if wps > 7.0: # Quá nhanh, có thể là model bị nhiễu sinh ra rác liên tục
        confidence -= 0.5
        has_uncertain = True
    elif wps < 0.3: # Quá chậm, 60s chỉ có vài từ
        confidence -= 0.2
        
    # 2. Hiện tượng lặp từ (CTC Stuttering/Loop)
    repeats = 0
    for i in range(1, len(words)):
        if words[i] == words[i-1]:
            repeats += 1
            
    repeat_ratio = repeats / len(words)
    if repeat_ratio > 0.3: # Lặp từ > 30% -> Chắc chắn lỗi
        confidence -= 0.6
        has_uncertain = True
    elif repeat_ratio > 0.15:
        confidence -= 0.3
        has_uncertain = True
        
    confidence = round(max(0.0, min(1.0, confidence)), 3)
    return confidence, has_uncertain


# ── Cell 6: Resume logic + PART split ────────────────────────────────────────
# Sort để đảm bảo thứ tự nhất quán giữa 2 notebooks
index_df = index_df.sort_values("filename").reset_index(drop=True)

# Chia index theo PART (A = nửa đầu, B = nửa sau)
mid = len(index_df) // 2
if PART == "A":
    part_df = index_df.iloc[:mid].copy()
else:
    part_df = index_df.iloc[mid:].copy()

logger.info(f"PART {PART}: {len(part_df):,} files (index {0 if PART=='A' else mid} → {mid if PART=='A' else len(index_df)})")

processed_filenames = set()
if OUTPUT_CSV.exists():
    df_existing = pd.read_csv(OUTPUT_CSV)
    processed_filenames = set(df_existing["filename"].astype(str))
    logger.info(f"Resume: {len(processed_filenames):,} files đã xử lý trước đó.")

rows_to_process = part_df[
    ~part_df["filename"].isin(processed_filenames)
].to_dict("records")

logger.info(f"Còn lại cần xử lý: {len(rows_to_process):,} / {len(part_df):,} files.")

# ── Cell 6: Core function — xử lý 1 file ─────────────────────────────────────
import warnings

def process_single_file(vocals_path: str, filename: str = None) -> dict:
    """
    Xử lý 1 file vocals.wav bằng ChunkFormer.
    """
    if filename is None:
        filename = Path(vocals_path).parent.name + ".mp4"

    duration_s = get_audio_duration(vocals_path)
    
    if duration_s < 0.5:
        return {
            "filename"      : filename,
            "raw_transcript": "",
            "asr_confidence": 0.0,
            "has_uncertain" : False,
            "duration_s"    : 0.0,
            "status"        : "load_failed",
        }

    try:
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Decode bằng ChunkFormer
            # Tham số chunk_size và context_size phù hợp với pre-trained config
            transcription = model.endless_decode(
                audio_path=vocals_path,
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
                total_batch_duration=14400,
                return_timestamps=False
            )

        # Chuẩn hoá output thành string
        if isinstance(transcription, list):
            raw_transcript = " ".join([str(t) for t in transcription])
        else:
            raw_transcript = str(transcription)
            
        raw_transcript = raw_transcript.strip()
        
        # CTC models thường ít hallucination rác như Whisper
        # Mặc định tự tin nếu có transcript
        confidence, has_uncertain = calculate_confidence(raw_transcript, duration_s)

        return {
            "filename": filename, 
            "raw_transcript": raw_transcript,
            "asr_confidence": confidence,
            "has_uncertain": has_uncertain,
            "duration_s": round(duration_s, 2),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"ASR thất bại [{filename}]: {e}")
        return {
            "filename"      : filename,
            "raw_transcript": "",
            "asr_confidence": 0.0,
            "has_uncertain" : False,
            "duration_s"    : 0.0,
            "status"        : "asr_failed",
        }


# ── Cell 6.5: TEST — Chạy thử 1 file bất kỳ ─────────────────────────────────
# ↓↓↓ TRUYỀN ĐƯỜNG DẪN FILE WAV VÀO ĐÂY ↓↓↓
TEST_WAV_PATH = "/kaggle/input/tiktok-vocal-audio/.../vocals.wav"

result = process_single_file(TEST_WAV_PATH)

print("=" * 70)
print(f"🧪 TEST: {result['filename']}")
print("=" * 70)
print(f"  Status     : {result['status']}")
print(f"  Confidence : {result['asr_confidence']:.3f}")
print(f"  Duration   : {result['duration_s']:.1f}s")
print(f"  Uncertain  : {result['has_uncertain']}")
print(f"  Transcript :")
print(f"  {result['raw_transcript']}")
print("=" * 70)

# ── Cell 7: Main extraction loop ──────────────────────────────────────────────
results_buffer = []
stats = {"success": 0, "load_failed": 0, "asr_failed": 0}
t_start = time.time()

for idx, row in enumerate(tqdm(rows_to_process, desc=f"ASR Extraction PART-{PART}")):
    filename    = row["filename"]
    vocals_path = row["vocals_path"]

    result = process_single_file(vocals_path, filename)

    # Update stats
    stats[result["status"]] = stats.get(result["status"], 0) + 1
    results_buffer.append(result)

    # Log ETA sau mỗi file
    elapsed_so_far = time.time() - t_start
    avg_per_file   = elapsed_so_far / (idx + 1)
    eta_min        = avg_per_file * (len(rows_to_process) - idx - 1) / 60
    logger.info(
        f"[{idx+1}/{len(rows_to_process):,} ({(idx+1)/len(rows_to_process)*100:.2f}%)] "
        f"| {filename} | {avg_per_file:.2f}s/file | ETA: {eta_min:.1f} min "
        f"| Success={stats['success']} LoadFail={stats['load_failed']} ASRFail={stats['asr_failed']}"
    )

    # Checkpoint save mỗi SAVE_EVERY files
    if len(results_buffer) >= SAVE_EVERY:
        df_buf = pd.DataFrame(results_buffer)
        write_header = not OUTPUT_CSV.exists()
        df_buf.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
        results_buffer = []

# Flush buffer còn lại
if results_buffer:
    df_buf = pd.DataFrame(results_buffer)
    write_header = not OUTPUT_CSV.exists()
    df_buf.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)

elapsed = time.time() - t_start


# ── Cell 8: Summary ───────────────────────────────────────────────────────────
df_final = pd.read_csv(OUTPUT_CSV)

logger.info("=" * 60)
logger.info(f"✅ Success     : {stats['success']:,}")
logger.info(f"❌ Load failed : {stats['load_failed']:,}")
logger.info(f"❌ ASR failed  : {stats['asr_failed']:,}")
logger.info(f"⏱️  Total time  : {elapsed / 60:.1f} min")
logger.info(f"📄 Rows in CSV : {len(df_final):,}")

success_df = df_final[df_final["status"] == "success"]
if len(success_df) > 0:
    logger.info(f"📊 Avg confidence   : {success_df['asr_confidence'].mean():.3f}")
    logger.info(f"⚠️  Has uncertain    : {success_df['has_uncertain'].sum():,} files")
    logger.info(f"📏 Avg duration     : {success_df['duration_s'].mean():.1f}s")
    logger.info(f"📝 Avg transcript   : {success_df['raw_transcript'].str.len().mean():.0f} chars")

logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save /kaggle/working/asr_transcripts/ → Kaggle Dataset 'tiktok-asr-transcripts'")
logger.info("  2. Attach Dataset đó vào NB-ASR-2 (asr_nb2_llm_postprocess.py)")
logger.info("=" * 60)


# ── Cell 9: [Optional] WER Benchmark trên CommonVoice VI ─────────────────────
"""
Đánh giá WER của PhoWhisper-large trên CommonVoice Vietnamese test set.
Chạy cell này riêng để có baseline WER, không cần run cùng với extraction loop.

!pip install jiwer datasets -q

import jiwer
from datasets import load_dataset

N_SAMPLES = 200   # Lấy 200 samples để estimate nhanh

cv_vi = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    "vi",
    split=f"test[:{N_SAMPLES}]",
    trust_remote_code=True
)

references  = []
hypotheses  = []

for item in tqdm(cv_vi, desc="WER Benchmark"):
    audio_array = item["audio"]["array"]
    ref_text    = item["sentence"]

    with torch.inference_mode():
        result = asr_pipeline(
            audio_array.astype(np.float32),
            generate_kwargs={"language": "vi", "task": "transcribe", "temperature": 0.0}
        )

    hyp_text = result.get("text", "").strip()
    references.append(ref_text)
    hypotheses.append(hyp_text)

word_error_rate = jiwer.wer(references, hypotheses)
char_error_rate = jiwer.cer(references, hypotheses)
logger.info(f"WER (CommonVoice VI, {N_SAMPLES} samples): {word_error_rate:.4f}")
logger.info(f"CER (CommonVoice VI, {N_SAMPLES} samples): {char_error_rate:.4f}")
"""
