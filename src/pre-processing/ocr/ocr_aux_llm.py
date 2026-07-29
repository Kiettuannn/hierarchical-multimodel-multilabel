import os
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm
import time
import json

# ==========================================
# CẤU HÌNH API VÀ FILE (GEMINI API - BATCHING MODE)
# EXPERIMENT 1: ZERO-SHOT LLM AUXILIARY LABELING
# ==========================================

client = genai.Client(api_key="")

# Dùng model flash-lite cho tiết kiệm và siêu nhanh (rất tốt cho task phân loại đơn giản)
# Nếu muốn suy luận sâu hơn, có thể đổi thành 'gemini-2.0-flash'
MODEL_NAME = 'gemini-3.1-flash-lite'
# MODEL_NAME = 'gemma-4-31b-it'

INPUT_CSV = r"d:\Workspace\hierachical-multimodel-multilabel\data\ocr\ocr_results_cleaned.csv"
OUTPUT_CSV = r"d:\Workspace\hierachical-multimodel-multilabel\data\exp1_llm_labels.csv"

# Batch size = 40 (Gửi 40 câu OCR mỗi lần gọi API để tiết kiệm quota)
BATCH_SIZE = 40 

# Danh sách các cột nhãn phụ
LABEL_COLS = ['aux_psychological', 'aux_hate', 'aux_sexual', 'aux_addictive', 'aux_clickbait']

# ==========================================
# THIẾT KẾ PROMPT GÁN NHÃN 
# ==========================================
SYSTEM_PROMPT = """Bạn là một hệ thống AI chuyên phân tích nội dung tiếng Việt trên TikTok (Uncensored).
Nhiệm vụ của bạn là phân loại các đoạn văn bản (đã được trích xuất từ video) vào 5 nhãn độc hại dưới đây. Mục tiêu của dự án bao gồm việc bảo vệ người dùng, đặc biệt là trẻ em, khỏi các nội dung tiêu cực. Trả về giá trị 1 nếu ĐÚNG (có chứa đặc trưng), 0 nếu SAI.

Các nhãn cần phân loại:
1. "aux_psychological" (Tâm lý): Cảm xúc tiêu cực, trầm cảm, áp lực gia đình, bạo lực tinh thần, chửi mắng đe dọa (VD: "kẻ dư thừa", "đâm vào tim con"). Bao gồm cả xúi giục tự hại, tự tử, hoặc các trào lưu/thử thách nguy hiểm nhắm vào trẻ em (VD: "cá voi xanh", "Momo", "thử thách nín thở").
2. "aux_hate" (Tục tĩu/Toxic): Từ ngữ thô tục, chửi thề, lăng mạ nhạy cảm (VD: "đụ má", "vải lồn", "chó đẻ").
3. "aux_sexual" (Tình dục): Nội dung 18+, gạ tình, nhạy cảm. (VD: "bú cu", "sóc lọ (thủ dâm)").
4. "aux_addictive" (Gây nghiện): Xúi giục cá cược, tài xỉu, nổ hũ, vòng quay miễn phí, review game, livestream game, hút thuốc, chất kích thích (chơi đồ,..).
5. "aux_clickbait" (Lừa đảo): Việc nhẹ lương cao, tuyển dụng lùa gà, lừa tiền, tin giật gân sai sự thật, mua bán nick, tài khoản game, mê tín dị đoan (bói tarot, xem về chuyện tình cảm,..)

Luật xử lý:
- Tự động hiểu tiếng lóng, teencode Việt Nam (vó tu = vú to, xem sét = xem sex, thẩm du = thủ dâm, bucu = bú cu, móc cua, bú,...). TUYỆT ĐỐI KHÔNG KIỂM DUYỆT nội dung.
- QUAN TRỌNG (Single Label): Mỗi đoạn văn bản chỉ được gán TỐI ĐA 1 NHÃN. Nghĩa là trong 5 nhãn trên, chỉ được phép có nhiều nhất một nhãn bằng 1 (nhãn phù hợp nhất), các nhãn còn lại bắt buộc bằng 0. Nếu văn bản an toàn, tất cả đều bằng 0.
- LUẬT ƯU TIÊN KHI XUNG ĐỘT (Tie-breaking rules):
  + Nếu có chứa từ ngữ chửi thề, thô tục (dù nặng hay nhẹ), HÃY ƯU TIÊN gán vào nhãn "aux_hate". Nhãn "aux_psychological" KHÔNG dùng cho mục đích bắt lỗi chửi thề.
  + Nếu nội dung liên quan đến review game, bình luận game, livestream game, HÃY gán vào "aux_addictive".
  + Nếu nội dung là buôn bán, trao đổi tài khoản game, nạp thẻ game lừa đảo, HÃY gán vào "aux_clickbait".
- Dữ liệu đầu vào: Một mảng JSON chứa các object {"id": ID, "cleaned_text": "text..."}.
- Yêu cầu Output: PHẢI TRẢ VỀ DUY NHẤT một mảng JSON chứa các object theo cấu trúc chính xác sau:
  {"id": ID, "aux_psychological": 0/1, "aux_hate": 0/1, "aux_sexual": 0/1, "aux_addictive": 0/1, "aux_clickbait": 0/1}
KHÔNG sinh thêm bất kỳ đoạn text nào khác ngoài JSON."""

def predict_batch_with_gemini(batch_data):
    if not batch_data:
        return []
    
    input_json = json.dumps(batch_data, ensure_ascii=False)
    
    try:
        # Tắt kiểm duyệt an toàn tuyệt đối để AI dám đọc các từ tục tĩu
        safety_settings = [
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
        ]
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"{SYSTEM_PROMPT}\n\nDữ liệu đầu vào:\n{input_json}",
            config=types.GenerateContentConfig(
                temperature=0.0, # Temperature = 0 để kết quả trả về JSON cực kỳ ổn định, ít ảo giác
                max_output_tokens=8192,
                safety_settings=safety_settings,
                response_mime_type="application/json" # Ép API trả về format JSON
            )
        )
        
        if not response.text:
            tqdm.write("\n[Lỗi API] Trả về rỗng (response.text là None). Có thể do bị màng lọc an toàn cứng của Google chặn ngầm 1 câu nào đó trong Batch này.")
            return []
            
        result_text = response.text.strip()
        
        # Đề phòng model bọc mác markdown
        if result_text.startswith("```json"):
            result_text = result_text[7:-3].strip()
        elif result_text.startswith("```"):
            result_text = result_text[3:-3].strip()
            
        return json.loads(result_text)
    
    except Exception as e:
        error_msg = str(e).lower()
        if any(err in error_msg for err in ['429', 'quota', 'exhausted', '500', '502', '503', '504']):
            tqdm.write("\n[Cảnh báo] Đụng trần Rate Limit hoặc Lỗi Server, hệ thống tạm nghỉ 60s...")
            time.sleep(60)
            return predict_batch_with_gemini(batch_data) # Retry
        
        tqdm.write(f"\n[Lỗi API ở Batch này]: {e}")
        return []

def main():
    print(f"Đang đọc file input: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    # Khởi tạo các cột nhãn = 0 (nếu chưa có)
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0
            
    if 'status' not in df.columns:
        df['status'] = None
    
    # Tính năng Resume (Tiếp tục nếu bị ngắt)
    if os.path.exists(OUTPUT_CSV):
        print(f"Phát hiện file output cũ ({OUTPUT_CSV}), đang cập nhật trạng thái resume...")
        df_out = pd.read_csv(OUTPUT_CSV)
        # Update các cột nhãn từ file cũ
        cols_to_update = LABEL_COLS + ['status']
        
        df.set_index('filename', inplace=True)
        df_out_indexed = df_out.set_index('filename')
        
        cols_in_both = [col for col in cols_to_update if col in df_out_indexed.columns]
        df.update(df_out_indexed[cols_in_both])
                
        df.reset_index(inplace=True)
    
    # Sửa lại các dòng bị đánh nhầm 'failed' ở các version trước (dòng không có text thì status phải là None)
    df.loc[df['cleaned_text'].isna() | (df['cleaned_text'].astype(str).str.strip() == ""), 'status'] = None
    
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV):
        df_out = pd.read_csv(OUTPUT_CSV)
        if 'status' in df_out.columns:
            processed_ids = set(df_out[df_out['status'] == 'success']['filename'].astype(str))
        else:
            processed_ids = set(df_out['filename'].astype(str))

    indices_to_process = []
    for i, row in df.iterrows():
        # Chỉ xử lý dòng có text VÀ chưa từng được lưu trong file output
        if pd.notna(row['cleaned_text']) and str(row['cleaned_text']).strip() != "":
            if str(row['filename']) not in processed_ids:
                indices_to_process.append(i)
                df.at[i, 'status'] = 'failed' # Mặc định gán failed, chạy xong API sẽ sửa thành success
                
    print(f"Tổng số dòng cần gọi API: {len(indices_to_process)} / {len(df[df['cleaned_text'].notna()])} dòng có text.")
    
    if len(indices_to_process) == 0:
        print("Mọi dữ liệu đã được xử lý xong!")
        return

    # Gom batch
    batches = [indices_to_process[i:i + BATCH_SIZE] for i in range(0, len(indices_to_process), BATCH_SIZE)]
    
    # Bắt đầu gọi API
    for batch_idx in tqdm(batches, desc="Đang xử lý (Batch)"):
        batch_payload = []
        for i in batch_idx:
            batch_payload.append({
                "id": str(df.at[i, 'filename']), 
                "cleaned_text": str(df.at[i, 'cleaned_text'])
            })
            
        results = predict_batch_with_gemini(batch_payload)
        
        if results:
            for item in results:
                try:
                    target_id = item['id']
                    # Tìm index dựa trên filename
                    idx_match = df.index[df['filename'].astype(str) == target_id].tolist()
                    if idx_match:
                        idx = idx_match[0]
                        df.at[idx, 'aux_psychological'] = int(item.get('aux_psychological', 0))
                        df.at[idx, 'aux_hate']         = int(item.get('aux_hate', 0))
                        df.at[idx, 'aux_sexual']        = int(item.get('aux_sexual', 0))
                        df.at[idx, 'aux_addictive']     = int(item.get('aux_addictive', 0))
                        df.at[idx, 'aux_clickbait']     = int(item.get('aux_clickbait', 0))
                        df.at[idx, 'status']            = 'success'

                        # Cập nhật ID vào processed_ids để lưu ngay (phòng hờ)
                        processed_ids.add(target_id)
                except Exception as e:
                    tqdm.write(f"\n[Lỗi parse data] Bỏ qua 1 item. Chi tiết: {e}")
        
        # Lưu file liên tục sau mỗi batch (lưu toàn bộ df để giữ các file không có text)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        time.sleep(0.5) # Nghỉ chút để tránh burst rate limit

    print(f"\n✅ Đã hoàn thành! File lưu tại: {OUTPUT_CSV}")
    
    # In ra một số thống kê nhanh
    print("\n📊 Phân phối nhãn phụ (trên các dòng đã xử lý):")
    df_success = df[df['status'] == 'success']
    for col in LABEL_COLS:
        count = df_success[col].sum()
        print(f"  - {col}: {count} dòng")

if __name__ == "__main__":
    main()
