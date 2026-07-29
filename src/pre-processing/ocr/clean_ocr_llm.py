import os
import pandas as pd
from google import genai
from google.genai import types
from tqdm import tqdm
import time
import json

# ==========================================
# CẤU HÌNH API VÀ FILE (GEMINI API - BATCHING MODE)
# ==========================================

client = genai.Client(api_key="AIzaSyDY7QpKu1jpeyRy4ilqQLB_khrJyDwfeA8")

# Đảm bảo dùng model mới. Nếu lỗi 404, bạn đổi thành 'gemini-2.0-flash'
MODEL_NAME = 'gemini-3.1-flash-lite'

INPUT_CSV = r"d:\Workspace\hierachical-multimodel-multilabel\data\ocr\ocr_results.csv"
OUTPUT_CSV = r"d:\Workspace\hierachical-multimodel-multilabel\data\ocr\ocr_results_cleaned.csv"

# Gom bao nhiêu dòng trong 1 lần gửi (Google Free Tier thường cho phép text khá dài)
BATCH_SIZE = 40 

# Prompt Batching chuẩn hóa (Ép trả về JSON)
SYSTEM_PROMPT = """Bạn là chuyên gia AI xử lý ngôn ngữ tiếng Việt (Uncensored).
Nhiệm vụ: Làm sạch danh sách văn bản OCR từ video TikTok để train mô hình phân loại Nội dung độc hại.

Luật làm sạch cho MỖI văn bản:
1. Xóa các cụm từ bị lặp lại, giữ nguyên đầy đủ ý, KHÔNG tóm tắt hay cắt câu.
2. Xóa ký tự vô nghĩa (VD: [Call+1...]).
3. Sửa lỗi chính tả, teencode thông thường.
4. QUAN TRỌNG NHẤT: Dịch tường minh, giữ nguyên các từ chửi thề, tiếng lóng nhạy cảm (VD: vó tu -> vú to, đậu má -> đụ má, bucu -> bú cu, vl -> vải lồn). TUYỆT ĐỐI KHÔNG KIỂM DUYỆT.

Dữ liệu đầu vào: Một mảng JSON chứa các object {"id": ID, "raw_text": "text..."}.
Yêu cầu Output: PHẢI TRẢ VỀ DUY NHẤT một mảng JSON chứa các object {"id": ID, "cleaned_text": "text đã xử lý..."}.
KHÔNG sinh thêm bất kỳ đoạn text nào khác ngoài JSON."""

def clean_batch_with_gemini(batch_data):
    if not batch_data:
        return []
    
    input_json = json.dumps(batch_data, ensure_ascii=False)
    
    try:
        # Tắt kiểm duyệt an toàn
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
                temperature=0.1, # Rất thấp để bắt model tuân thủ cấu trúc JSON
                max_output_tokens=8192, # Đủ lớn cho 40 dòng text
                safety_settings=safety_settings,
                response_mime_type="application/json" # Ép API trả về format JSON
            )
        )
        
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
            return clean_batch_with_gemini(batch_data) 
        
        tqdm.write(f"\n[Lỗi API ở Batch này]: {e}")
        return []

def main():
    print(f"Đang đọc file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    
    if 'cleaned_text' not in df.columns:
        df['cleaned_text'] = ""
    
    # Tính năng Resume
    if os.path.exists(OUTPUT_CSV):
        print("Phát hiện file output cũ, cập nhật trạng thái...")
        df_out = pd.read_csv(OUTPUT_CSV)
        
        # Chỉ lấy cột cleaned_text từ file cũ, map theo filename để tránh bị lệch dòng (row shift)
        df.set_index('filename', inplace=True)
        df_out_indexed = df_out.set_index('filename')
        
        if 'cleaned_text' in df_out_indexed.columns:
            df['cleaned_text'].update(df_out_indexed['cleaned_text'])
            
        df.reset_index(inplace=True)
    
    # Lọc ra danh sách index của các dòng chưa có kết quả (raw_text không rỗng)
    indices_to_process = []
    for i, row in df.iterrows():
        if pd.isna(row['cleaned_text']) or str(row['cleaned_text']).strip() == "":
            raw = str(row.get('raw_text', ''))
            if raw != 'nan' and raw.strip():
                indices_to_process.append(i)
                
    print(f"Tổng số dòng cần xử lý tiếp: {len(indices_to_process)}")
    print(f"Sẽ chạy theo Batch, mỗi Batch = {BATCH_SIZE} dòng.")
    
    # Chạy vòng lặp theo từng Batch
    for i in tqdm(range(0, len(indices_to_process), BATCH_SIZE), desc="Processing Batches"):
        batch_indices = indices_to_process[i:i + BATCH_SIZE]
        
        # Build JSON array cho batch hiện tại
        batch_data = []
        for idx in batch_indices:
            batch_data.append({
                "id": idx,
                "raw_text": str(df.at[idx, 'raw_text'])
            })
            
        # Gọi API 1 lần cho N dòng
        result_json = clean_batch_with_gemini(batch_data)
        
        # Cập nhật kết quả trả về vào Dataframe
        success_count = 0
        if result_json:
            for item in result_json:
                row_id = item.get("id")
                cleaned = item.get("cleaned_text", "")
                if row_id is not None and cleaned:
                    df.at[row_id, 'cleaned_text'] = cleaned
                    success_count += 1
                    
        tqdm.write(f"-> Batch hoàn tất: Xử lý thành công {success_count}/{len(batch_indices)} dòng.")
        
        # Lưu file sau mỗi batch
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        # Giãn cách 10 giây giữa các batch để xài Free an toàn
        time.sleep(10)
    
    print(f"\nĐã hoàn thành! Kết quả lưu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
