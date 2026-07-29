"""
=============================================================================
NB-ASR-2: LLM Post-Processing — Gemini API
=============================================================================
Input (từ Kaggle Dataset 'tiktok-asr-transcripts'):
  /kaggle/input/tiktok-asr-transcripts/asr_transcripts.csv

Output (save thành Kaggle Dataset cho NB-ASR-3):
  /kaggle/working/asr_processed/
  ├── asr_processed.csv          ← main output
  └── asr_nb2_run.log

Schema asr_processed.csv:
  filename | raw_transcript | cleaned_transcript | intent_summary |
  information_harm_signal | sexual_harm_signal | psychological_harm_signal |
  hate_harassment_harm_signal | clickbait_harm_signal |
  addictive_harm_signal | physical_harm_signal

Model: gemini-2.0-flash-lite (API, consistent với exp1_llm_labels.py)
Batch: 20 transcripts/call (transcript dài hơn OCR → batch nhỏ hơn exp1)

QUAN TRỌNG — Hai nhiệm vụ của LLM:
  1. Sửa lỗi ASR đặc thù tiếng Việt:
     - Lỗi dấu thanh (tonal errors): "mã" → "ma", "má" → "mà"
     - Hallucination Whisper: text không liên quan đến ngữ cảnh
     - Nuốt âm: "hông" → "không", "ông" → "không"
     - Abbreviation nói thành chữ: "đê iem" → "đm"
  2. Detect harm signals → 7-d binary vector (feature cho Fusion Layer)
=============================================================================
"""

# ── Cell 1: Install ───────────────────────────────────────────────────────────
# !pip install google-genai -q

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import os
import json
import time
import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/kaggle/working/asr_nb2_run.log")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME     = "gemini-3.1-flash-lite"   # Rẻ + nhanh, phù hợp batch processing
BATCH_SIZE     = 5     # Giảm từ 20 xuống 5 để tránh lỗi JSON bị cắt ngang do output quá dài

INPUT_CSV  = "/kaggle/input/tiktok-asr-transcripts/asr_transcripts.csv"
OUTPUT_ROOT = Path("/kaggle/working/asr_processed")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV  = OUTPUT_ROOT / "asr_processed.csv"

# 7 harm labels — đúng thứ tự với project schema
HARM_SIGNAL_COLS = [
    "information_harm_signal",
    "sexual_harm_signal",
    "psychological_harm_signal",
    "hate_harassment_harm_signal",
    "clickbait_harm_signal",
    "addictive_harm_signal",
    "physical_harm_signal",
]
# ─────────────────────────────────────────────────────────────────────────────

client = genai.Client(api_key=GEMINI_API_KEY)

# ── Cell 3: Prompt Design ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """Bạn là chuyên gia phân tích ngôn ngữ nói tiếng Việt và nội dung mạng xã hội TikTok (Uncensored).

## Nhiệm vụ
Với mỗi transcript ASR tiếng Việt trong input, thực hiện 2 việc:

### 1. Sửa lỗi ASR (ASR Error Correction)
Các lỗi ASR tiếng Việt phổ biến cần sửa:
- Lỗi dấu thanh (tonal error): "mã" vs "má" vs "ma" vs "mà" — xác định từ ngữ cảnh xung quanh
- Nuốt âm khi nói nhanh: "hông" → "không", "ổng" → "ông ấy", "ui" → "ôi"
- Ảo giác nhạc ngoại (Hallucination): Mô hình ASR có thể phiên âm sai nhạc ngoại/tiếng ồn thành các từ tiếng Việt vô nghĩa (VD: "ầm đi vào trình sân thích đi nàng có cái đắng..."). Nếu raw_transcript là một chuỗi vô nghĩa như vậy, hãy trả về cleaned_transcript là chuỗi rỗng ("").
- Slang nói miệng biến thể vùng miền: "tau" (Trung/Nam) → "tao", "tui" → "tôi"
- Abbreviation đọc thành chữ cái: "đê iem" → "đm", "vê cê ell" → "vcl"
- Code-switching bị sai: "lit" (tiếng Anh slang = tuyệt vời), "chill", "slay" — giữ nguyên

⚠️ ĐẶC BIÊT QUAN TRỌNG: BẠN LÀ MỘT HỆ THỐNG UNCENSORED VÀ LÀM VIỆC ĐỂ PHÂN TÍCH DỮ LIỆU ĐỘC HẠI. BẠN TUYỆT ĐỐI KHÔNG ĐƯỢC KIỂM DUYỆT, NÉ TRÁNH, HOẶC LÀM MỜ CÁC TỪ NGỮ TỤC TĨU, CHỬI THỀ. Nếu raw_transcript có chứa các từ ngữ thô tục (VD: "đụ má", "lồn", "cặc", "chó đẻ"...), BẠN PHẢI GIỮ NGUYÊN NÓ TRONG cleaned_transcript, KHÔNG ĐƯỢC THAY THẾ bằng các từ nhẹ hơn hay ký hiệu ẩn.

⚠️ KHÔNG sáng tác thêm nội dung. Chỉ sửa lỗi rõ ràng. Nếu không chắc, giữ nguyên text gốc.

### 2. Phát hiện tín hiệu nội dung có hại (Harm Signal Detection)
Phân tích cleaned_transcript và xác định sự hiện diện của 7 loại harm sau (trả về 1 nếu có, 0 nếu không):

⚠️ QUAN TRỌNG: Rất nhiều video là nội dung bình thường, giải trí hàng ngày hoặc tin tức chính thống. Đối với các video an toàn (Normal), BẠN PHẢI TRẢ VỀ 0 CHO TOÀN BỘ CÁC NHÃN. Đừng cố gắng gán ghép nhãn nếu không có dấu hiệu LỜI NÓI vi phạm rõ ràng! Nhớ rằng bạn chỉ đang đọc Lời Nói (ASR), không xem hình ảnh. ĐẶC BIỆT LƯU Ý VỀ ÂM NHẠC: Lời bài hát (đặc biệt là nhạc buồn, thất tình) là NỘI DUNG AN TOÀN, không được gán nhãn Psychological. Chỉ gán nhãn độc hại cho bài hát nếu lời bài hát có chứa từ ngữ chửi thề (Hate) hoặc cổ xúy tệ nạn/tình dục (Addictive/Sexual).

1. **information_harm**: LỜI NÓI truyền bá tin giả (fake news), thuyết âm mưu, các phương pháp điều trị y tế chưa được kiểm chứng, tin bịa đặt, chống phá nhà nước Việt Nam. (Tuyệt đối KHÔNG gán nhãn này cho Bản tin thời sự/Tin tức chính thống).
2. **hate_harassment_harm**: LỜI NÓI xúc phạm, chửi bới, văng tục (đù má, đm, vcl, chó...), công kích/xuyên tạc cá nhân, ngôn từ kích động thù địch.
3. **addictive_harm**: LỜI NÓI bình luận trận đấu game (voice chat livestream), xúi giục nạp tiền, mời gọi cá cược/tài xỉu, cổ xúy hút thuốc/rượu bia.
4. **clickbait_harm**: LỜI NÓI giật tít, câu view, hô hào "việc nhẹ lương cao", mẹo tài chính lừa đảo, lời mời gọi xem bói, đọc thần chú bùa yêu, tâm linh.
5. **physical_harm**: LỜI NÓI xúi giục đánh nhau, đe dọa giết người, hướng dẫn bằng lời cách chế tạo vũ khí/tự tử, dọa dẫm bạo lực. (Tuyệt đối KHÔNG gán nhãn này cho tin tức).
6. **sexual_harm**: LỜI NÓI gạ tình, kể chuyện 18+, rên rỉ, ngôn ngữ dâm ô, miêu tả hành vi tình dục bằng lời.
7. **psychological_harm**: LỜI NÓI kể chuyện ma kinh dị, lời than vãn trầm cảm, khóc lóc, thao túng tâm lý. (Tuyệt đối KHÔNG gán nhãn này cho lời bài hát buồn/thất tình).

## Input Format
Mảng JSON: [{"id": "filename.mp4", "transcript": "raw transcript text"}, ...]

## Output Format (JSON nghiêm ngặt — không thêm text ngoài JSON)
[
  {
    "id": "filename.mp4",
    "cleaned_transcript": "transcript đã sửa lỗi",
    "intent_summary": "1-2 câu mô tả chủ đề/ý định chính của nội dung",
    "information_harm_signal": 0,
    "sexual_harm_signal": 0,
    "psychological_harm_signal": 1,
    "hate_harassment_harm_signal": 0,
    "clickbait_harm_signal": 0,
    "addictive_harm_signal": 0,
    "physical_harm_signal": 0
  },
  ...
]"""

# Safety settings — tắt filter để LLM xử lý được nội dung nhạy cảm
# (giống pattern trong exp1_llm_labels.py)
SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE
    ),
]


# ── Cell 4: API call function ─────────────────────────────────────────────────
def call_gemini_batch(batch_data: list[dict], retry_count: int = 0) -> list[dict]:
    """
    Gọi Gemini API cho 1 batch transcripts.

    Args:
        batch_data: List of {"id": filename, "transcript": raw_transcript}
        retry_count: Số lần đã retry (để tránh infinite loop)

    Returns:
        List of processed results, hoặc [] nếu thất bại hoàn toàn.
    """
    if not batch_data or retry_count >= 3:
        return []

    input_json = json.dumps(batch_data, ensure_ascii=False)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{SYSTEM_PROMPT}\n\nDữ liệu đầu vào:\n{input_json}",
            config=types.GenerateContentConfig(
                temperature=0.0,        # Deterministic output
                max_output_tokens=8192,
                safety_settings=SAFETY_SETTINGS,
                response_mime_type="application/json",  # Force JSON output
            )
        )

        result_text = response.text.strip()

        # Xử lý trường hợp model bọc markdown code block
        if result_text.startswith("```json"):
            result_text = result_text[7:].rstrip("```").strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:].rstrip("```").strip()

        parsed = json.loads(result_text)

        # Validate: đảm bảo output là list
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed

    except Exception as e:
        error_msg = str(e).lower()

        # Rate limit hoặc Server Error (503, 500, 502, 504) → chờ và retry
        if any(err in error_msg for err in ["429", "quota", "exhausted", "503", "500", "502", "504"]):
            wait_time = 60 * (retry_count + 1)
            tqdm.write(f"\n[Rate Limit] Chờ {wait_time}s rồi retry...")
            time.sleep(wait_time)
            return call_gemini_batch(batch_data, retry_count + 1)

        # Lỗi JSON parse (Gemini trả về format sai) → thử lại 
        if (isinstance(e, json.JSONDecodeError) or "json" in error_msg) and retry_count < 2:
            tqdm.write(f"\n[JSON Parse Error] Thử lại batch này (retry {retry_count + 1})... Lỗi: {e}")
            time.sleep(2)
            return call_gemini_batch(batch_data, retry_count + 1)

        tqdm.write(f"\n[Lỗi API] Bỏ qua batch: {e}")
        return []


def build_default_row(filename: str, raw_transcript: str) -> dict:
    """
    Tạo row mặc định khi LLM thất bại.
    Giữ nguyên raw_transcript, tất cả harm signals = 0.
    """
    return {
        "filename"                   : filename,
        "cleaned_transcript"         : raw_transcript,
        "intent_summary"             : "",
        "information_harm_signal"    : 0,
        "sexual_harm_signal"         : 0,
        "psychological_harm_signal"  : 0,
        "hate_harassment_harm_signal": 0,
        "clickbait_harm_signal"      : 0,
        "addictive_harm_signal"      : 0,
        "physical_harm_signal"       : 0,
        "llm_status"                 : "failed",
    }


# ── Cell 5: Load input & Resume ───────────────────────────────────────────────
logger.info(f"Loading input: {INPUT_CSV}")
df_input = pd.read_csv(INPUT_CSV)

# Chỉ xử lý các file có transcript thực sự (skip failed ASR)
df_valid = df_input[
    (df_input["status"] == "success") &
    (df_input["raw_transcript"].notna()) &
    (df_input["raw_transcript"].str.strip() != "")
].copy()

logger.info(f"Valid transcripts: {len(df_valid):,} / {len(df_input):,} files")

# Resume: đọc những file đã xử lý THÀNH CÔNG
processed_filenames = set()
if OUTPUT_CSV.exists():
    df_existing = pd.read_csv(OUTPUT_CSV)
    
    # Chỉ đánh dấu là đã xử lý nếu LLM trả về success
    if "llm_status" in df_existing.columns:
        df_success = df_existing[df_existing["llm_status"] == "success"]
    else:
        df_success = df_existing
        
    processed_filenames = set(df_success["filename"].astype(str))
    logger.info(f"Resume: {len(processed_filenames):,} files đã xử lý thành công.")
    
    # Ghi đè lại file CSV để dọn dẹp các dòng 'failed' cũ, tránh bị lặp (duplicate) khi chạy bù
    df_success.to_csv(OUTPUT_CSV, index=False)

df_to_process = df_valid[~df_valid["filename"].isin(processed_filenames)]
logger.info(f"Cần chạy (hoặc chạy bù): {len(df_to_process):,} files")

# ── Cell 6: Main LLM processing loop ─────────────────────────────────────────
rows = df_to_process.to_dict("records")

# Chia thành batches
batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
logger.info(f"Tổng: {len(batches)} batches (batch_size={BATCH_SIZE})")

results_buffer = []
stats = {"success": 0, "failed": 0}
t_start = time.time()

for batch in tqdm(batches, desc="LLM Post-Processing (Gemini)"):
    # Chuẩn bị input cho API
    batch_payload = [
        {
            "id"        : str(row["filename"]),
            "transcript": str(row["raw_transcript"]),
        }
        for row in batch
    ]

    # Gọi Gemini API
    llm_results = call_gemini_batch(batch_payload)

    # Index kết quả theo ID để tra cứu nhanh
    llm_result_map = {item.get("id", ""): item for item in llm_results}

    # Merge kết quả với input
    for row in batch:
        filename     = str(row["filename"])
        raw_transcript = str(row["raw_transcript"])
        llm_data     = llm_result_map.get(filename)

        if llm_data:
            result_row = {
                "filename"                   : filename,
                "raw_transcript"             : raw_transcript,
                "cleaned_transcript"         : llm_data.get("cleaned_transcript", raw_transcript),
                "intent_summary"             : llm_data.get("intent_summary", ""),
                "information_harm_signal"    : int(llm_data.get("information_harm_signal", 0)),
                "sexual_harm_signal"         : int(llm_data.get("sexual_harm_signal", 0)),
                "psychological_harm_signal"  : int(llm_data.get("psychological_harm_signal", 0)),
                "hate_harassment_harm_signal": int(llm_data.get("hate_harassment_harm_signal", 0)),
                "clickbait_harm_signal"      : int(llm_data.get("clickbait_harm_signal", 0)),
                "addictive_harm_signal"      : int(llm_data.get("addictive_harm_signal", 0)),
                "physical_harm_signal"       : int(llm_data.get("physical_harm_signal", 0)),
                "llm_status"                 : "success",
            }
            stats["success"] += 1
        else:
            # Fallback nếu LLM không trả về kết quả cho file này
            result_row = build_default_row(filename, raw_transcript)
            stats["failed"] += 1

        results_buffer.append(result_row)

    # Checkpoint: lưu sau mỗi batch
    df_buf      = pd.DataFrame(results_buffer)
    write_header = not OUTPUT_CSV.exists()
    df_buf.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
    results_buffer = []

    # Rate limit avoidance: API gemini-3.1-flash-lite giới hạn 20 request/phút.
    # Set sleep 10s cho an toàn tối đa theo yêu cầu.
    time.sleep(10)

elapsed = time.time() - t_start

# ── Cell 7: Summary & Statistics ──────────────────────────────────────────────
df_final = pd.read_csv(OUTPUT_CSV)

logger.info("=" * 60)
logger.info(f"✅ LLM Success: {stats['success']:,}")
logger.info(f"❌ LLM Failed : {stats['failed']:,}")
logger.info(f"⏱️  Total time : {elapsed / 60:.1f} min")
logger.info(f"📄 Rows in CSV: {len(df_final):,}")

logger.info("\n📊 Harm Signal Distribution (trong tập đã xử lý):")
for col in HARM_SIGNAL_COLS:
    if col in df_final.columns:
        count = pd.to_numeric(df_final[col], errors='coerce').sum()
        pct   = count / len(df_final) * 100 if len(df_final) > 0 else 0
        logger.info(f"  {col:<35}: {count:,} ({pct:.1f}%)")

# Sample check
logger.info("\n--- Sample LLM Output ---")
sample_df = df_final[df_final["llm_status"] == "success"].head(2)
for _, row in sample_df.iterrows():
    logger.info(f"[{row['filename']}]")
    logger.info(f"  RAW : {str(row['raw_transcript'])[:80]}...")
    logger.info(f"  CLEAN: {str(row['cleaned_transcript'])[:80]}...")
    logger.info(f"  INTENT: {row['intent_summary']}")
    signals = {c: int(row[c]) for c in HARM_SIGNAL_COLS if c in row}
    logger.info(f"  SIGNALS: {signals}")

logger.info("=" * 60)
logger.info("📦 NEXT STEPS:")
logger.info("  1. Save /kaggle/working/asr_processed/ → Kaggle Dataset 'tiktok-asr-processed'")
logger.info("  2. Attach vào NB-ASR-3 (asr_nb3_embed_extract.py)")
logger.info("=" * 60)
