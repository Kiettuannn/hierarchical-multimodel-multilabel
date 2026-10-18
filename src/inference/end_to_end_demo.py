# %% [markdown]
# # 🎬 End-to-End Video Inference — Hierarchical Multimodal Cascade
#
# **Thứ tự xử lý (tuần tự, quản lý VRAM nghiêm ngặt):**
# 1. Vision Branch: 8 frames → Gemma4 API (scene_description + visual_cues) → PhoBERT + SigLIP2 → 1536-d
# 2. ASR Branch:   Demucs (vocals) → ChunkFormer → Gemini API (clean + 7-d priors) → PhoBERT → 768-d + 7-d
# 3. OCR Branch:   PaddleOCR + VietOCR → Gemini API (clean + 5-d priors) → PhoBERT → 768-d + 5-d
# 4. Audio Branch: Demucs (no_vocals) → CLAP → 512-d
# 5. Fusion:       Concat → 3596-d → Cascade (Stage1 → Stage2)

# %% [markdown]
# ## ⚙️ Cell 1 — Install & Configuration

# %%
# !pip install torch transformers paddlepaddle-gpu paddleocr librosa google-genai rapidfuzz vietocr -q
# !pip install git+https://github.com/facebookresearch/demucs.git -q
# !pip install chunkformer -q

import os, gc, sys, json, time, logging, warnings, re, traceback
import numpy as np
import torch
import torch.nn as nn
import librosa
import cv2
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from google import genai
from google.genai import types
from rapidfuzz import fuzz

warnings.filterwarnings("ignore")

# ── Logger ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")

# ── PATHS ──────────────────────────────────────────────────────────
VIDEO_PATH      = "/kaggle/input/your-demo-video/demo.mp4"    # ← Thay đường dẫn video
STAGE1_CKPT     = "/kaggle/input/stage-1-dataset/stage1_checkpoints/best_model.pt"
STAGE2_CKPT     = "/kaggle/input/stage-2-dataset/stage2_checkpoints/best_model.pt"
THRESHOLDS_PATH = "/kaggle/input/stage-2-dataset/stage2_checkpoints/best_thresholds.json"
WORK_DIR        = Path("/kaggle/working/inference_tmp")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ── API KEYS (mỗi branch dùng 1 key riêng để tránh rate limit) ────
GEMMA4_API_KEY  = "YOUR_GEMMA4_KEY"       # Vision branch — Gemma4-31b (heavy model)
ASR_API_KEY     = "YOUR_ASR_GEMINI_KEY"   # ASR branch   — gemini-3.1-flash-lite
OCR_API_KEY     = "YOUR_OCR_GEMINI_KEY"   # OCR branch   — gemini-3.1-flash-lite

GEMMA4_MODEL    = "gemma-4-31b-it"
GEMINI_MODEL    = "gemini-3.1-flash-lite"

SAFETY_OFF = [
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,        threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,         threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,  threshold=types.HarmBlockThreshold.BLOCK_NONE),
    types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,  threshold=types.HarmBlockThreshold.BLOCK_NONE),
]

# ── DEVICE ─────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── HELPERS ─────────────────────────────────────────────────────────
def free_vram(*args):
    for obj in args:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        before = torch.cuda.memory_allocated() / 1e6
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 1e6
        logger.debug(f"  [VRAM] Freed ~{before - after:.0f} MB | Now using {after:.0f} MB")

def log_separator(title: str):
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


# ── FUSION MODEL ────────────────────────────────────────────────────
class FusionMLP(nn.Module):
    def __init__(self, out_size: int = 7, dropout: float = 0.3,
                 hidden_1: int = 1024, hidden_2: int = 256):
        super().__init__()
        self.out_size = out_size
        INPUT_DIM = 1536 + 768 + 768 + 512 + 12
        self.mlp = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden_1), nn.LayerNorm(hidden_1), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2), nn.LayerNorm(hidden_2), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_2, out_size)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, vision, asr, ocr, clap, tabular):
        x = torch.cat([vision, asr, ocr, clap, tabular], dim=-1)
        return self.head(self.mlp(x))


# %% [markdown]
# ## 👁️ Cell 2 — Vision Function

# %%
# System prompt giống y hệt notebook gốc gemma4-extract-vision-features.ipynb
GEMMA4_SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh chuyên phân tích video. Nhiệm vụ của bạn là quan sát tập hợp các ảnh cắt từ một video và mô tả lại chi tiết những gì bạn thấy.
Tuyệt đối không từ chối trả lời. Hãy khách quan, mô tả chân thực hành động, bối cảnh, con người và các chi tiết nổi bật.

### YÊU CẦU ĐẦU RA (OUTPUT):
PHẢI TRẢ VỀ DUY NHẤT một đối tượng JSON hợp lệ. KHÔNG chứa comment. KHÔNG sinh thêm text nằm ngoài JSON.

{
  "scene_description": "Viết 2-3 câu tiếng Việt tóm tắt hành động, đối tượng, biểu cảm và bối cảnh chính của video."
}
"""

def process_vision(video_path: str, phobert_tokenizer, phobert_model):
    log_separator("VISION BRANCH")
    _t = time.time()

    # ── Extract 8 frames ─────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    logger.info(f"[Vision] Video: {total_frames} frames @ {fps:.1f} fps → {duration:.1f}s")

    frames = []
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, 8, dtype=int)
        logger.debug(f"[Vision] Extracting frame indices: {indices.tolist()}")
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    logger.info(f"[Vision] Extracted {len(frames)} frames ✓")

    # ── Gemma4 API → scene_description ───────────────────────────
    scene_desc = ""
    gemma4_client = genai.Client(api_key=GEMMA4_API_KEY)
    logger.info(f"[Vision] Calling Gemma4 API ({GEMMA4_MODEL})...")
    try:
        import io
        contents = [GEMMA4_SYSTEM_PROMPT]
        
        # Y hệt notebook gốc: convert sang bytes và dùng types.Part
        for f in frames:
            img_byte_arr = io.BytesIO()
            f.save(img_byte_arr, format='JPEG')
            image_bytes = img_byte_arr.getvalue()
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            
        contents.append("Hãy phân tích chuỗi ảnh này và trả về kết quả JSON cho toàn bộ video.")

        response = gemma4_client.models.generate_content(
            model=GEMMA4_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                safety_settings=SAFETY_OFF,
            )
        )
        raw_text = response.text
        logger.debug(f"[Vision] Gemma4 raw response: {raw_text}")

        if raw_text:
            try:
                parsed = json.loads(raw_text)
                scene_desc = parsed.get("scene_description", "").strip()
                logger.info(f"[Vision] scene_description: '{scene_desc}'")
            except Exception as e:
                scene_desc = f"<JSON Parse Error: {e} | Raw: {raw_text}>"
                logger.warning(f"[Vision] JSON parse failed: {e}")
        else:
            scene_desc = "<Gemma4 returned empty text/None>"
            logger.warning("[Vision] Gemma4 returned None/empty.")
            logger.warning(f"[Vision] FULL RESPONSE DUMP: {response}")
            logger.warning("[Vision] Falling back to zero text emb.")

    except Exception as e:
        scene_desc = f"<API Error: {e}>"
        logger.error(f"[Vision] Gemma4 API error: {e}")
        logger.debug(traceback.format_exc())


    # ── SigLIP2 → Image Embedding (768-d) ────────────────────────
    image_emb = np.zeros(768, dtype=np.float32)
    if frames:
        SIGLIP_MODEL = "google/siglip2-base-patch16-224"
        logger.info(f"[Vision] Loading SigLIP2: {SIGLIP_MODEL}")
        siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL)
        siglip_model = AutoModel.from_pretrained(SIGLIP_MODEL).to(DEVICE)
        siglip_model.eval()
        with torch.no_grad():
            inputs = siglip_processor(images=frames, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            vision_out = siglip_model.vision_model(**inputs)
            image_emb = vision_out.pooler_output.mean(dim=0).cpu().numpy().astype(np.float32)
        logger.info(f"[Vision] SigLIP2 image_emb: shape={image_emb.shape}, norm={np.linalg.norm(image_emb):.4f}")
        free_vram(siglip_model, siglip_processor)

    # ── PhoBERT → Text Embedding từ scene_desc (768-d) ───────────
    text_vision_emb = np.zeros(768, dtype=np.float32)
    if scene_desc:
        logger.info(f"[Vision] Embedding scene_description with PhoBERT...")
        with torch.no_grad():
            inputs = phobert_tokenizer(scene_desc, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = phobert_model(**inputs)
            text_vision_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype(np.float32)
        logger.info(f"[Vision] text_vision_emb: norm={np.linalg.norm(text_vision_emb):.4f}")
    else:
        logger.warning("[Vision] scene_desc is empty → text_vision_emb = zeros(768)")

    vision_emb = np.concatenate([image_emb, text_vision_emb])
    logger.info(f"[Vision] ✅ Done in {time.time()-_t:.1f}s | vision_emb shape={vision_emb.shape}, norm={np.linalg.norm(vision_emb):.4f}")
    return vision_emb, scene_desc, frames


# %% [markdown]
# ## 🗣️ Cell 3 — ASR Function

# %%
def process_asr(video_path: str, demucs_out: Path, phobert_tokenizer, phobert_model):
    log_separator("ASR BRANCH")
    _t = time.time()

    asr_emb = np.zeros(768, dtype=np.float32)
    asr_prior = np.zeros(7, dtype=np.float32)
    raw_transcript = ""
    cleaned_transcript = ""

    # ── Demucs: tách vocals ───────────────────────────────────────
    logger.info(f"[ASR] Running Demucs (two-stems=vocals)...")
    logger.info(f"[ASR]   Input : {video_path}")
    logger.info(f"[ASR]   Output: {demucs_out}")

    # Thêm -d cpu để tránh lỗi "cublasLtCreate" (xung đột CUDA context giữa subprocess và main process)
    ret = os.system(f'python -m demucs -d cpu --two-stems=vocals -o "{demucs_out}" "{video_path}" --mp3')
    logger.info(f"[ASR] Demucs exit code: {ret}")

    # Log tất cả files được tạo ra
    all_output = list(demucs_out.rglob("*"))
    logger.debug(f"[ASR] Demucs output tree ({len(all_output)} entries):")
    for f in sorted(all_output):
        size = f.stat().st_size if f.is_file() else "-"
        logger.debug(f"[ASR]   {'FILE' if f.is_file() else 'DIR '} {f} ({size} bytes)")

    vocals_files = list(demucs_out.rglob("vocals.wav")) + list(demucs_out.rglob("vocals.mp3"))
    logger.info(f"[ASR] vocals_files found: {vocals_files}")
    VOCALS_PATH = str(vocals_files[0]) if vocals_files else None

    if VOCALS_PATH is None:
        logger.warning("[ASR] No vocals file found! Demucs may have failed. Skipping ASR.")
        return asr_emb, asr_prior, raw_transcript, cleaned_transcript

    # ── ChunkFormer → raw_transcript ─────────────────────────────
    logger.info(f"[ASR] Loading vocals: {VOCALS_PATH}")
    try:
        y, sr = librosa.load(VOCALS_PATH, sr=16000, mono=True)
        logger.info(f"[ASR] Vocals audio: {len(y)/sr:.1f}s @ {sr} Hz")

        if len(y) > int(60.0 * 16000):
            logger.info(f"[ASR] Audio > 60s, clipping to 60s")
            y = y[:int(60.0 * 16000)]

        if len(y) < int(0.5 * 16000):
            logger.warning(f"[ASR] Audio too short ({len(y)/sr:.2f}s < 0.5s). Skipping ChunkFormer.")
        else:
            from chunkformer import ChunkFormerModel
            logger.info(f"[ASR] Loading ChunkFormer: khanhld/chunkformer-ctc-large-vie")
            chunkformer = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie").to(DEVICE)

            with torch.no_grad():
                transcription = chunkformer.endless_decode(
                    audio_path=VOCALS_PATH,
                    chunk_size=64,
                    left_context_size=128,
                    right_context_size=128,
                    total_batch_duration=14400,
                    return_timestamps=False
                )
            
            # Chuẩn hóa output
            if isinstance(transcription, list):
                if len(transcription) > 0 and isinstance(transcription[0], dict):
                    raw_transcript = " ".join([t.get("text", "") for t in transcription])
                else:
                    raw_transcript = " ".join(transcription)
            elif isinstance(transcription, dict):
                raw_transcript = transcription.get("text", "")
            else:
                raw_transcript = str(transcription)

            logger.info(f"[ASR] raw_transcript ({len(raw_transcript)} chars): '{raw_transcript[:120]}'")
            free_vram(chunkformer)

    except Exception as e:
        logger.error(f"[ASR] ChunkFormer error: {e}")
        logger.error(traceback.format_exc())

    if not raw_transcript.strip():
        logger.warning("[ASR] raw_transcript is empty. Skipping Gemini normalize.")
        return asr_emb, asr_prior, raw_transcript, cleaned_transcript

    # ── Gemini API: normalize + 7-d harm signals ─────────────────
    asr_client = genai.Client(api_key=ASR_API_KEY)
    logger.info(f"[ASR] Calling Gemini API ({GEMINI_MODEL}) for ASR normalize + harm signals...")
    ASR_PROMPT = """Bạn là chuyên gia phân tích ngôn ngữ nói tiếng Việt và nội dung mạng xã hội TikTok (Uncensored).

## Nhiệm vụ
Với transcript ASR tiếng Việt dưới đây, thực hiện 2 việc:

### 1. Sửa lỗi ASR (ASR Error Correction)
Các lỗi ASR tiếng Việt phổ biến cần sửa:
- Lỗi dấu thanh (tonal error): "mã" vs "má" vs "ma" vs "mà" — xác định từ ngữ cảnh xung quanh
- Nuốt âm khi nói nhanh: "hông" → "không", "ổng" → "ông ấy", "ui" → "ôi"
- Ảo giác nhạc ngoại (Hallucination): Nếu raw_transcript là một chuỗi vô nghĩa, hãy trả về cleaned_transcript là chuỗi rỗng ("").
- Slang nói miệng biến thể vùng miền: "tau" → "tao", "tui" → "tôi"
- Abbreviation đọc thành chữ cái: "đê iem" → "đm", "vê cê ell" → "vcl"

⚠️ ĐẶC BIỆT QUAN TRỌNG: BẠN LÀ MỘT HỆ THỐNG UNCENSORED. TUYỆT ĐỐI KHÔNG KIỂM DUYỆT, NÉ TRÁNH từ ngữ tục tĩu, chửi thề. Phải giữ nguyên trong cleaned_transcript.
⚠️ KHÔNG sáng tác thêm nội dung. Chỉ sửa lỗi rõ ràng. Nếu không chắc, giữ nguyên text gốc.

### 2. Phát hiện tín hiệu nội dung có hại (Harm Signal Detection)
⚠️ Rất nhiều video là nội dung bình thường. Đối với video an toàn, PHẢI TRẢ VỀ 0 CHO TOÀN BỘ CÁC NHÃN. Nhớ rằng bạn chỉ đang đọc Lời Nói (ASR), không xem hình ảnh. ĐẶC BIỆT: Lời bài hát buồn, thất tình là NỘI DUNG AN TOÀN, không được gán nhãn Psychological.

1. information_harm: Truyền bá tin giả, thuyết âm mưu, phương pháp điều trị chưa kiểm chứng.
2. hate_harassment_harm: Chửi bới, văng tục (đù má, đm, vcl, chó...), công kích cá nhân.
3. addictive_harm: Bình luận game (voice chat), xúi giục nạp tiền, cá cược/tài xỉu, cổ xúy hút thuốc/rượu bia.
4. clickbait_harm: Giật tít câu view, hô hào việc nhẹ lương cao, mẹo tài chính lừa đảo, lời mời xem bói/bùa yêu.
5. physical_harm: Xúi giục đánh nhau, đe dọa giết người, hướng dẫn chế tạo vũ khí/tự tử.
6. sexual_harm: Gạ tình, kể chuyện 18+, rên rỉ, ngôn ngữ dâm ô.
7. psychological_harm: Kể chuyện ma kinh dị, than vãn trầm cảm, thao túng tâm lý. KHÔNG gán cho lời bài hát buồn.

## Output Format (JSON nghiêm ngặt)
{"cleaned_transcript": "...", "intent_summary": "1-2 câu mô tả chủ đề/ý định chính",
 "information_harm_signal": 0, "sexual_harm_signal": 0, "psychological_harm_signal": 0,
 "hate_harassment_harm_signal": 0, "clickbait_harm_signal": 0,
 "addictive_harm_signal": 0, "physical_harm_signal": 0}"""

    try:
        resp = asr_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"{ASR_PROMPT}\n\nTranscript:\n{raw_transcript}",
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048,
                                               safety_settings=SAFETY_OFF, response_mime_type="application/json")
        )
        raw_resp = resp.text
        logger.debug(f"[ASR] Gemini raw response: {raw_resp}")
        parsed = json.loads(raw_resp)
        cleaned_transcript = parsed.get("cleaned_transcript", "")
        intent_summary = parsed.get("intent_summary", "")
        asr_prior = np.array([
            float(parsed.get("information_harm_signal", 0)),
            float(parsed.get("sexual_harm_signal", 0)),
            float(parsed.get("psychological_harm_signal", 0)),
            float(parsed.get("hate_harassment_harm_signal", 0)),
            float(parsed.get("clickbait_harm_signal", 0)),
            float(parsed.get("addictive_harm_signal", 0)),
            float(parsed.get("physical_harm_signal", 0)),
        ], dtype=np.float32)
        logger.info(f"[ASR] cleaned_transcript ({len(cleaned_transcript)} chars): '{cleaned_transcript[:100]}'")
        logger.info(f"[ASR] intent_summary: '{intent_summary}'")
        logger.info(f"[ASR] asr_prior: {asr_prior.tolist()}")

    except Exception as e:
        logger.error(f"[ASR] Gemini normalize error: {e}")
        logger.error(traceback.format_exc())
        cleaned_transcript = raw_transcript
        logger.warning(f"[ASR] Fallback: using raw_transcript as cleaned_transcript")

    # ── PhoBERT → ASR Embedding (768-d) ──────────────────────────
    if cleaned_transcript.strip():
        logger.info(f"[ASR] Embedding cleaned_transcript with PhoBERT...")
        with torch.no_grad():
            inputs = phobert_tokenizer(cleaned_transcript, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = phobert_model(**inputs)
            asr_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype(np.float32)
        logger.info(f"[ASR] asr_emb: norm={np.linalg.norm(asr_emb):.4f}")
    else:
        logger.warning(f"[ASR] cleaned_transcript empty → asr_emb = zeros(768)")

    logger.info(f"[ASR] ✅ Done in {time.time()-_t:.1f}s")
    return asr_emb, asr_prior, raw_transcript, cleaned_transcript


# %% [markdown]
# ## 📝 Cell 4 — OCR Engines Init + Function
#
# ⚠️ PaddleOCR (paddlex backend) chỉ được khởi tạo 1 LẦN duy nhất mỗi Python session.
# Đặt init ở đây để tránh RuntimeError: "PDX has already been initialized".

# %%
TIKTOK_UI_PATTERNS = [
    r'(?i)tik\s*tok', r'(?i)j[a-z]*tik\s*tok', r'^[A-Za-z0-9@_\.]{5,30}$', r'^\d{1,7}[KMB]?$',
    r'^[\u266a\u266b\U0001F3B5\U0001F3B6\u2764\U0001F44D\U0001F4AC\U0001F517]+$'
]
TIKTOK_UI_KEYWORDS = {'tiktok', 'follow', 'following', 'followers', 'like', 'share', 'comment', 'duet'}

def is_ui_text(text: str) -> bool:
    t = text.strip()
    if len(t) < 2: return True
    if t.lower() in TIKTOK_UI_KEYWORDS: return True
    for p in TIKTOK_UI_PATTERNS:
        if re.match(p, t): return True
    if sum(c.isdigit() for c in t) / len(t) > 0.7: return True
    has_vi = bool(re.search(r'[\u00c0-\u024f\u1e00-\u1eff]', t, re.IGNORECASE))
    if not has_vi and len(t) < 20: return True
    return False

def crop_text_region(frame_bgr, poly_points):
    try:
        pts = np.array(poly_points, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        H, W = frame_bgr.shape[:2]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if (x2 - x) < 5 or (y2 - y) < 5: return None
        return Image.fromarray(cv2.cvtColor(frame_bgr[y:y2, x:x2], cv2.COLOR_BGR2RGB))
    except:
        return None


# ── Khởi tạo OCR engines 1 lần (module-level) ──────────────────────
logger.info("[OCR-Init] Loading PaddleOCR (PP-OCRv5 server detect)...")
from paddleocr import PaddleOCR as PaddleOCRCls
ocr_engine = PaddleOCRCls(
    use_textline_orientation=True, use_doc_orientation_classify=False, use_doc_unwarping=False,
    lang="latin", text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec", rec_batch_num=6,
)

logger.info("[OCR-Init] Loading VietOCR (vgg_transformer recognize)...")
from vietocr.tool.predictor import Predictor as VietOCRPredictor
from vietocr.tool.config import Cfg as VietOCRCfg
viet_cfg = VietOCRCfg.load_config_from_name('vgg_transformer')
viet_cfg['device'] = 'cuda:0' if str(DEVICE) == 'cuda' else 'cpu'
viet_cfg['predictor']['beamsearch'] = False
viet_ocr = VietOCRPredictor(viet_cfg)
logger.info("[OCR-Init] OCR engines ready ✓")


def process_ocr(frames: list, ocr_engine, viet_ocr, phobert_tokenizer, phobert_model):
    log_separator("OCR BRANCH")
    _t = time.time()

    ocr_emb = np.zeros(768, dtype=np.float32)
    ocr_prior = np.zeros(5, dtype=np.float32)
    raw_text = ""
    cleaned_ocr = ""

    # Chỉ chạy trên 3 frames (đầu/giữa/cuối)
    OCR_NUM_FRAMES = 3
    ocr_frame_indices = np.linspace(0, len(frames) - 1, OCR_NUM_FRAMES, dtype=int).tolist()
    logger.info(f"[OCR] Running on {OCR_NUM_FRAMES} frames (indices: {ocr_frame_indices})")

    raw_results = []
    for fi, frame_idx in enumerate(ocr_frame_indices):
        frame_pil = frames[int(frame_idx)]
        frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        logger.debug(f"[OCR] Processing frame {fi} (global idx {frame_idx})")

        try:
            paddle_result = ocr_engine.predict(frame_bgr)
        except Exception as e:
            logger.warning(f"[OCR] PaddleOCR error on frame {fi}: {e}")
            continue

        if not paddle_result:
            logger.debug(f"[OCR] Frame {fi}: PaddleOCR returned empty")
            continue

        res = paddle_result[0]
        rec_scores = res.get("rec_scores", [])
        rec_polys  = res.get("rec_polys", [])
        logger.debug(f"[OCR] Frame {fi}: {len(rec_polys)} regions detected")

        frame_texts = []
        for score, poly in zip(rec_scores, rec_polys):
            det_conf = float(score)
            if det_conf < 0.6:
                continue
            try:
                box_pts = [[int(p[0]), int(p[1])] for p in poly]
            except:
                continue
            pil_crop = crop_text_region(frame_bgr, box_pts)
            if pil_crop is None:
                continue
            try:
                vi_text, vi_conf = viet_ocr.predict(pil_crop, return_prob=True)
            except Exception as e:
                logger.debug(f"[OCR] VietOCR predict error: {e}")
                continue

            vi_text = str(vi_text).strip()
            vi_conf = float(vi_conf)
            logger.debug(f"[OCR]   region: '{vi_text}' (det={det_conf:.3f}, rec={vi_conf:.3f})")

            if vi_text and vi_conf >= 0.5:
                frame_texts.append({"text": vi_text, "vi_conf": vi_conf, "det_conf": det_conf})

        if frame_texts:
            logger.debug(f"[OCR] Frame {fi}: {len(frame_texts)} valid texts")
            raw_results.append({"frame_idx": fi, "texts": frame_texts})

    free_vram(ocr_engine, viet_ocr)

    logger.info(f"[OCR] Raw results: {len(raw_results)} frames with text")

    # ── Filter watermarks ─────────────────────────────────────────
    deduped_texts = []
    seen_pool: list = []  # dùng fuzzy dedup
    for item in raw_results:
        for t in item["texts"]:
            clean_t = t["text"]
            if is_ui_text(clean_t):
                logger.debug(f"[OCR] Filtered UI text: '{clean_t}'")
                continue
            # Fuzzy dedup: bỏ nếu >85% giống với bất kỳ text đã thêm
            is_dup = any(fuzz.ratio(clean_t.lower(), s) > 85 for s in seen_pool)
            if is_dup:
                logger.debug(f"[OCR] Fuzzy-dedup removed: '{clean_t}'")
                continue
            seen_pool.append(clean_t.lower())
            deduped_texts.append(clean_t)

    raw_text = " ".join(deduped_texts)
    logger.info(f"[OCR] After filter+dedup: {len(deduped_texts)} tokens")
    logger.info(f"[OCR] raw_text: '{raw_text}'")

    if not raw_text.strip():
        logger.warning("[OCR] raw_text is empty. Skipping Gemini calls.")
        logger.info(f"[OCR] ✅ Done in {time.time()-_t:.1f}s")
        return ocr_emb, ocr_prior, raw_text, cleaned_ocr

    # ── Gemini API: OCR Normalize ─────────────────────────────────
    ocr_client = genai.Client(api_key=OCR_API_KEY)
    logger.info(f"[OCR] Calling Gemini API ({GEMINI_MODEL}) for OCR normalize...")
    OCR_NORM_PROMPT = """Bạn là chuyên gia AI xử lý ngôn ngữ tiếng Việt (Uncensored).
Nhiệm vụ: Làm sạch văn bản OCR từ video TikTok.

Luật làm sạch:
1. Xóa các cụm từ bị lặp lại, giữ nguyên đầy đủ ý, KHÔNG tóm tắt hay cắt câu.
2. Xóa ký tự vô nghĩa (VD: [Call+1...]).
3. Sửa lỗi chính tả, teencode thông thường.
4. QUAN TRỌNG NHẤT: Dịch tường minh, giữ nguyên các từ chửi thề, tiếng lóng nhạy cảm (VD: vó tu → vú to, đậu má → đụ má, bucu → bú cu, vl → vải lồn). TUYỆT ĐỐI KHÔNG KIỂM DUYỆT.

Output JSON: {"cleaned_text": "..."}"""
    try:
        resp = ocr_client.models.generate_content(
            model=GEMINI_MODEL, contents=f"{OCR_NORM_PROMPT}\n\nOCR text:\n{raw_text}",
            config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=2048,
                                               safety_settings=SAFETY_OFF, response_mime_type="application/json")
        )
        logger.debug(f"[OCR] Normalize raw response: {resp.text}")
        cleaned_ocr = json.loads(resp.text).get("cleaned_text", raw_text)
        logger.info(f"[OCR] cleaned_ocr: '{cleaned_ocr}'")
    except Exception as e:
        logger.error(f"[OCR] Gemini normalize error: {e}")
        logger.error(traceback.format_exc())
        cleaned_ocr = raw_text

    # ── Gemini API: OCR Aux Priors (5-d) ─────────────────────────
    logger.info(f"[OCR] Calling Gemini API ({GEMINI_MODEL}) for OCR aux priors...")
    OCR_AUX_PROMPT = """Bạn là một hệ thống AI chuyên phân tích nội dung tiếng Việt trên TikTok (Uncensored).
Nhiệm vụ: Phân loại văn bản OCR vào 5 nhãn độc hại. Trả về 1 nếu ĐÚNG, 0 nếu SAI.

Các nhãn:
1. "aux_psychological": Cảm xúc tiêu cực, trầm cảm, áp lực gia đình, bạo lực tinh thần. Bao gồm xúi giục tự hại, tự tử.
2. "aux_hate": Từ ngữ thô tục, chửi thề, lăng mạ nhạy cảm (VD: đụ má, vải lồn, chó đẻ).
3. "aux_sexual": Nội dung 18+, gạ tình, nhạy cảm (VD: bú cu, sóc lọ).
4. "aux_addictive": Xúi giục cá cược, tài xỉu, nổ hũ, vòng quay miễn phí, review game, hút thuốc.
5. "aux_clickbait": Việc nhẹ lương cao, tuyển dụng lùa gà, lừa tiền, tin giật gân sai sự thật, mua bán nick game, mê tín dị đoan.

Luật xử lý:
- QUAN TRỌNG (Single Label): Mỗi đoạn văn bản chỉ được gán TỐI ĐA 1 NHÃN bằng 1. Nếu an toàn, tất cả đều bằng 0.
- LUẬT ƯU TIÊN KHI XUNG ĐỘT:
  + Nếu có từ ngữ chửi thề/thô tục → ƯU TIÊN gán "aux_hate".
  + Nếu liên quan review game/bình luận game/livestream game → gán "aux_addictive".
  + Nếu buôn bán/trao đổi tài khoản game/nạp thẻ game lừa đảo → gán "aux_clickbait".

Output JSON: {"aux_psychological":0,"aux_hate":0,"aux_sexual":0,"aux_addictive":0,"aux_clickbait":0}"""
    try:
        resp = ocr_client.models.generate_content(
            model=GEMINI_MODEL, contents=f"{OCR_AUX_PROMPT}\n\nVăn bản OCR:\n{cleaned_ocr}",
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=256,
                                               safety_settings=SAFETY_OFF, response_mime_type="application/json")
        )
        logger.debug(f"[OCR] Aux priors raw response: {resp.text}")
        parsed = json.loads(resp.text)
        ocr_prior = np.array([
            float(parsed.get("aux_psychological", 0)), float(parsed.get("aux_hate", 0)),
            float(parsed.get("aux_sexual", 0)), float(parsed.get("aux_addictive", 0)),
            float(parsed.get("aux_clickbait", 0))
        ], dtype=np.float32)
        logger.info(f"[OCR] ocr_prior: {ocr_prior.tolist()}")
    except Exception as e:
        logger.error(f"[OCR] Gemini aux priors error: {e}")
        logger.error(traceback.format_exc())

    # ── PhoBERT → OCR Embedding (768-d) ──────────────────────────
    if cleaned_ocr.strip():
        logger.info(f"[OCR] Embedding cleaned_ocr with PhoBERT...")
        with torch.no_grad():
            inputs = phobert_tokenizer(cleaned_ocr, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = phobert_model(**inputs)
            ocr_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().astype(np.float32)
        logger.info(f"[OCR] ocr_emb: norm={np.linalg.norm(ocr_emb):.4f}")
    else:
        logger.warning("[OCR] cleaned_ocr empty → ocr_emb = zeros(768)")

    logger.info(f"[OCR] ✅ Done in {time.time()-_t:.1f}s")
    return ocr_emb, ocr_prior, raw_text, cleaned_ocr


# %% [markdown]
# ## 🎵 Cell 5 — Audio Function

# %%
def process_audio(demucs_out: Path):
    log_separator("AUDIO BRANCH")
    _t = time.time()

    audio_emb = np.zeros(512, dtype=np.float32)

    no_vocals_files = list(demucs_out.rglob("no_vocals.wav")) + list(demucs_out.rglob("no_vocals.mp3"))
    logger.info(f"[Audio] no_vocals_files found: {no_vocals_files}")
    NO_VOCALS_PATH = str(no_vocals_files[0]) if no_vocals_files else None

    if NO_VOCALS_PATH is None:
        logger.warning("[Audio] no_vocals file not found! Demucs output may be missing.")
        logger.warning(f"[Audio] Searched in: {demucs_out}")
        logger.warning(f"[Audio] All files in WORK_DIR: {list(demucs_out.rglob('*'))[:20]}")
        logger.info(f"[Audio] ✅ Done in {time.time()-_t:.1f}s (CLAP skipped, using zeros)")
        return audio_emb

    logger.info(f"[Audio] Loading no_vocals: {NO_VOCALS_PATH}")
    try:
        y, sr = librosa.load(NO_VOCALS_PATH, sr=48000, mono=True)
        logger.info(f"[Audio] No-vocals audio: {len(y)/sr:.1f}s @ {sr} Hz")

        if len(y) > int(60.0 * 48000):
            y = y[:int(60.0 * 48000)]

        if len(y) < int(0.1 * 48000):
            logger.warning(f"[Audio] Audio too short ({len(y)/sr:.2f}s), skipping CLAP.")
        else:
            from transformers import ClapModel, ClapProcessor
            CLAP_MODEL = "laion/larger_clap_general"
            logger.info(f"[Audio] Loading CLAP: {CLAP_MODEL}")
            clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)
            clap_model = ClapModel.from_pretrained(CLAP_MODEL).to(DEVICE)
            clap_model.eval()

            inputs = clap_processor(audios=y, sampling_rate=48000, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                pooled = clap_model.audio_model(**inputs).pooler_output
                projected = clap_model.audio_projection(pooled)
            audio_emb = projected.squeeze().cpu().numpy().astype(np.float32)
            logger.info(f"[Audio] audio_emb: shape={audio_emb.shape}, norm={np.linalg.norm(audio_emb):.4f}")
            free_vram(clap_model, clap_processor)

    except Exception as e:
        logger.error(f"[Audio] CLAP error: {e}")
        logger.error(traceback.format_exc())

    logger.info(f"[Audio] ✅ Done in {time.time()-_t:.1f}s")
    return audio_emb


# %% [markdown]
# ## 🚀 Cell 6 — Main Pipeline Execution

# %%
print("\n" + "═"*80)
print("🚀 BẮT ĐẦU CHẠY PIPELINE END-TO-END")
print(f"   Video: {VIDEO_PATH}")
print("═"*80)

T_TOTAL_START = time.time()
timing_report = {}
DEMUCS_OUT = WORK_DIR / "demucs_out"

# ── 0. Load PhoBERT (shared across Vision/ASR/OCR) ───────────────
_t0 = time.time()
print("\n[0/5] Load PhoBERT...")
logger.info("[Init] Loading PhoBERT: vinai/phobert-base-v2")
phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
phobert_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(DEVICE)
phobert_model.eval()
logger.info(f"[Init] PhoBERT loaded ✓ | VRAM used: {torch.cuda.memory_allocated()/1e6:.0f} MB")
timing_report["0_init"] = time.time() - _t0

# ── 1. Vision ─────────────────────────────────────────────────────
_t0 = time.time()
print("\n[1/5] Vision Branch...")
vision_emb, scene_desc, frames = process_vision(VIDEO_PATH, phobert_tokenizer, phobert_model)
timing_report["1_vision"] = time.time() - _t0
print(f"  👉 SCENE DESC : {scene_desc if scene_desc else '<Không có — Gemma4 trả rỗng>'}")
print(f"  👉 vision_emb : norm={np.linalg.norm(vision_emb):.4f}, zeros={np.sum(vision_emb==0)}/1536")

# ── 2. ASR ────────────────────────────────────────────────────────
_t0 = time.time()
print("\n[2/5] ASR Branch...")
asr_emb, asr_prior, raw_asr, cleaned_asr = process_asr(VIDEO_PATH, DEMUCS_OUT, phobert_tokenizer, phobert_model)
timing_report["2_asr"] = time.time() - _t0
print(f"  👉 ASR RAW     : '{raw_asr[:150]}'")
print(f"  👉 ASR CLEANED : '{cleaned_asr[:150]}'")
print(f"  👉 ASR PRIORS  : {asr_prior.tolist()}")
print(f"  👉 asr_emb     : norm={np.linalg.norm(asr_emb):.4f}, zeros={np.sum(asr_emb==0)}/768")

# ── 3. OCR ────────────────────────────────────────────────────────
_t0 = time.time()
print("\n[3/5] OCR Branch...")
ocr_emb, ocr_prior, raw_ocr, cleaned_ocr = process_ocr(frames, ocr_engine, viet_ocr, phobert_tokenizer, phobert_model)
timing_report["3_ocr"] = time.time() - _t0
free_vram(phobert_model, phobert_tokenizer)
print(f"  👉 OCR RAW     : '{raw_ocr}'")
print(f"  👉 OCR CLEANED : '{cleaned_ocr}'")
print(f"  👉 OCR PRIORS  : {ocr_prior.tolist()}")
print(f"  👉 ocr_emb     : norm={np.linalg.norm(ocr_emb):.4f}, zeros={np.sum(ocr_emb==0)}/768")

# ── 4. Audio ──────────────────────────────────────────────────────
_t0 = time.time()
print("\n[4/5] Audio Branch...")
audio_emb = process_audio(DEMUCS_OUT)
timing_report["4_audio"] = time.time() - _t0
print(f"  👉 audio_emb   : norm={np.linalg.norm(audio_emb):.4f}, zeros={np.sum(audio_emb==0)}/512")

# ── Sanity check: bao nhiêu branch có embedding non-zero? ─────────
n_nonzero = sum([
    np.linalg.norm(vision_emb) > 0.1,
    np.linalg.norm(asr_emb) > 0.1,
    np.linalg.norm(ocr_emb) > 0.1,
    np.linalg.norm(audio_emb) > 0.1,
])
logger.info(f"[Sanity] {n_nonzero}/4 branches have non-zero embeddings")
if n_nonzero < 2:
    logger.warning("[Sanity] ⚠️ Less than 2 branches have embeddings! Model output may be unreliable.")

# ── 5. Cascade Inference ──────────────────────────────────────────
_t0 = time.time()
print("\n[5/5] Cascade Inference...")

LABEL_NAMES = ["information_harm", "sexual_harm", "psychological_harm",
               "hate_harassment_harm", "clickbait_harm", "addictive_harm", "physical_harm"]

tabular_prior = np.concatenate([asr_prior, ocr_prior]).astype(np.float32)
logger.info(f"[Cascade] tabular_prior (12-d): {tabular_prior.tolist()}")

vision_t  = torch.tensor(vision_emb,    dtype=torch.float32).unsqueeze(0).to(DEVICE)
asr_t     = torch.tensor(asr_emb,       dtype=torch.float32).unsqueeze(0).to(DEVICE)
ocr_t     = torch.tensor(ocr_emb,       dtype=torch.float32).unsqueeze(0).to(DEVICE)
audio_t   = torch.tensor(audio_emb,     dtype=torch.float32).unsqueeze(0).to(DEVICE)
tabular_t = torch.tensor(tabular_prior, dtype=torch.float32).unsqueeze(0).to(DEVICE)

stage1_model = FusionMLP(out_size=1).to(DEVICE)
try:
    ckpt = torch.load(STAGE1_CKPT, map_location=DEVICE)
    stage1_model.load_state_dict(ckpt)
    logger.info("[Cascade] Stage 1 checkpoint loaded ✓")
except Exception as e:
    logger.error(f"[Cascade] Cannot load Stage 1 checkpoint: {e}")
    logger.error(traceback.format_exc())

stage1_model.eval()
with torch.no_grad():
    s1_logit = stage1_model(vision_t, asr_t, ocr_t, audio_t, tabular_t)
    p_harmful = torch.sigmoid(s1_logit).item()
logger.info(f"[Cascade] Stage 1 logit={s1_logit.item():.4f} → P(Harmful)={p_harmful:.4f}")
free_vram(stage1_model)

if p_harmful < 0.5:
    print("\n" + "="*60)
    print(f"✅  KẾT LUẬN: BÌNH THƯỜNG (Normal Content) | P={p_harmful:.4f}")
    print("="*60)
else:
    logger.info(f"[Cascade] P(Harmful)={p_harmful:.4f} ≥ 0.5 → Forward to Stage 2")

    stage2_model = FusionMLP(out_size=7).to(DEVICE)
    try:
        ckpt = torch.load(STAGE2_CKPT, map_location=DEVICE)
        stage2_model.load_state_dict(ckpt)
        logger.info("[Cascade] Stage 2 checkpoint loaded ✓")
    except Exception as e:
        logger.error(f"[Cascade] Cannot load Stage 2 checkpoint: {e}")

    try:
        with open(THRESHOLDS_PATH, "r") as f:
            thr_dict = json.load(f)
        thresholds = [thr_dict.get(lbl, 0.5) for lbl in LABEL_NAMES]
        logger.info(f"[Cascade] Thresholds: {dict(zip(LABEL_NAMES, thresholds))}")
    except Exception as e:
        logger.warning(f"[Cascade] Cannot load thresholds: {e}. Using 0.5 default.")
        thresholds = [0.5] * 7

    stage2_model.eval()
    with torch.no_grad():
        s2_logits = stage2_model(vision_t, asr_t, ocr_t, audio_t, tabular_t)
        p_labels = torch.sigmoid(s2_logits).squeeze().cpu().numpy()
    logger.info(f"[Cascade] Stage 2 logits: {s2_logits.squeeze().tolist()}")
    logger.info(f"[Cascade] Stage 2 probs : {p_labels.tolist()}")
    free_vram(stage2_model)

    print("\n" + "="*60)
    print(f"🚨  KẾT LUẬN: ĐỘC HẠI (Harmful Content) | P={p_harmful:.4f}")
    print("="*60)
    print("\n📋 Chi tiết từng nhãn:")
    violations = []
    for lbl, prob, thr in zip(LABEL_NAMES, p_labels, thresholds):
        is_violation = prob >= thr
        icon = "❌" if is_violation else "✅"
        print(f"  {icon}  {lbl:<28} P={prob:.3f}  (ngưỡng={thr:.2f})")
        if is_violation: violations.append(lbl)

    if not violations:
        violations = [LABEL_NAMES[int(np.argmax(p_labels))]]
        print(f"\n⚠️  Không nhãn nào vượt ngưỡng → Fallback: chọn nhãn cao nhất 👉 {violations[0]}")

    print("\n" + "="*60)
    print(f"🏷️  NHÃN VI PHẠM: {', '.join(violations)}")
    print("="*60)

timing_report["5_cascade"] = time.time() - _t0
total_elapsed = time.time() - T_TOTAL_START

# ── Feature Summary (In lại tổng hợp) ─────────────────────────────
print("\n" + "═"*80)
print("📊 TỔNG HỢP KẾT QUẢ TRÍCH XUẤT ĐẶC TRƯNG (FEATURE SUMMARY)")
print("═"*80)
print(f"👁️ VISION BRANCH:")
print(f"   - Scene Desc: {scene_desc if scene_desc else '<None>'}")
print(f"   - Embedding : norm={np.linalg.norm(vision_emb):.4f}")
print(f"\n🗣️ ASR BRANCH:")
print(f"   - Raw Text  : '{raw_asr}'")
print(f"   - Cleaned   : '{cleaned_asr}'")
print(f"   - Priors    : {asr_prior.tolist()}")
print(f"   - Embedding : norm={np.linalg.norm(asr_emb):.4f}")
print(f"\n📝 OCR BRANCH:")
print(f"   - Raw Text  : '{raw_ocr}'")
print(f"   - Cleaned   : '{cleaned_ocr}'")
print(f"   - Priors    : {ocr_prior.tolist()}")
print(f"   - Embedding : norm={np.linalg.norm(ocr_emb):.4f}")
print(f"\n🎵 AUDIO BRANCH:")
print(f"   - Embedding : norm={np.linalg.norm(audio_emb):.4f}")
print("═"*80)

# ── Timing Report ─────────────────────────────────────────────────
print("\n" + "═"*80)
print("⏱  BÁO CÁO THỜI GIAN (TIMING REPORT)")
print("═"*80)
stage_labels = {
    "0_init"    : "Load PhoBERT Model",
    "1_vision"  : "Vision Branch    (Gemma4 API + SigLIP2 + PhoBERT)",
    "2_asr"     : "ASR Branch       (Demucs + ChunkFormer + Gemini + PhoBERT)",
    "3_ocr"     : "OCR Branch       (PaddleOCR + VietOCR + Gemini×2 + PhoBERT)",
    "4_audio"   : "Audio Branch     (CLAP)",
    "5_cascade" : "Cascade Inference(Stage1 → Stage2)",
}
for key, label in stage_labels.items():
    t = timing_report.get(key, 0.0)
    bar = "█" * int(t / total_elapsed * 30) + "░" * (30 - int(t / total_elapsed * 30))
    print(f"  {label:<55} [{bar}] {t:6.1f}s  ({t/total_elapsed*100:.1f}%)")
print("═"*80)
print(f"  🏁 TỔNG THỜI GIAN: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} phút)")
print("═"*80)
