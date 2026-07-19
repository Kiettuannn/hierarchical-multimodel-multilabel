# Kế hoạch Triển khai Hệ thống Phân loại Video Độc hại (Hierarchical Multi-label)

Dựa trên yêu cầu và bối cảnh (context) về bài toán phân loại video TikTok độc hại đa phương thức (multimodal), kiến trúc cascade 2 tầng (Stage 1: Binary, Stage 2: Multi-label) là một lựa chọn tối ưu giúp giảm nhiễu (noise) từ video bình thường khi huấn luyện các nhãn độc hại chi tiết. 

Dưới đây là kế hoạch chi tiết cho từng cấu phần của hệ thống.

## Open Questions

> [!IMPORTANT]
> **Tỷ lệ chia dữ liệu:** Bạn muốn chia tập Train / Val / Test theo tỷ lệ nào? (Khuyến nghị: 70/15/15 hoặc 80/10/10 do dataset khá nhỏ ~6k videos).
> **Xử lý Out-of-Distribution (OOD):** Ở bước inference, nếu Stage 1 đoán là `harmful` (xác suất > 0.5) nhưng ở Stage 2 xác suất của cả 7 nhãn đều rất thấp (< ngưỡng, ví dụ < 0.5), hệ thống sẽ fallback như thế nào? (Gợi ý: Lấy nhãn có xác suất cao nhất (argmax) của Stage 2, hoặc gán 1 nhãn mặc định).
> **Threshold Tối ưu:** Ngưỡng phân loại (threshold) mặc định là 0.5. Bạn có muốn tích hợp bước tìm threshold tối ưu trên tập Validation (ví dụ dùng F1-Macro search) cho từng nhãn ở Stage 2 không?

## 1. Script Chia Dữ liệu (Data Spliting)

Mục tiêu: Đảm bảo phân phối nhãn (đặc biệt là các nhãn hiếm như `information_harm` và trường hợp đa nhãn) đồng đều giữa Train, Val, Test và ngăn chặn data leakage.

**Bước 1: Tạo nhãn `normal_content` và tính toán cột tổng hợp**
- Đọc file `data/label-origin.csv.csv`.
- Tạo thêm cột `normal_content`: Nếu tổng 7 cột harm == 0 thì `normal_content = 1`, ngược lại `= 0`.
- Ma trận phân tầng (Stratification matrix) sẽ bao gồm 8 cột (7 harm + 1 normal).

**Bước 2: Iterative Stratification**
- Sử dụng thuật toán `IterativeStratification` từ thư viện `scikit-multilearn` (skmultilearn). Thuật toán này phân bổ các dòng dựa trên tổ hợp nhãn, rất phù hợp cho multi-label.
- Thực hiện 2 lần split: `Toàn bộ -> Train / Temp` và `Temp -> Val / Test`.
- Tạo file chân lý (Single Source of Truth): `split_assignment.csv` chứa 2 cột: `filename`, `split` (mang giá trị 'train', 'val', 'test').

**Bước 3: Sinh CSV cho từng Stage**
- **Stage 1 (Binary CSV):**
  - Join `split_assignment.csv` với bảng gốc.
  - Tạo cột `is_harmful`: `1` nếu `normal_content == 0`, `0` nếu `normal_content == 1`.
  - Xuất 3 file: `stage1_train.csv`, `stage1_val.csv`, `stage1_test.csv` (chứa `filename`, `is_harmful`).
- **Stage 2 (Multi-label CSV):**
  - Lọc dữ liệu: Chỉ giữ lại các dòng có `is_harmful == 1` (chỉ dùng các video harmful).
  - Xuất 3 file: `stage2_train.csv`, `stage2_val.csv`, `stage2_test.csv` (chứa `filename` và 7 cột nhãn độc hại).

## 2. Pipeline Training 2 Tầng (Cascade + Warm-start)

### Stage 1: Phân loại nhị phân (Binary Classification)
- **Mục tiêu:** Tách biệt video `normal_content` và `harmful`.
- **Dataset / DataLoader:** Load từ `stage1_train.csv`.
- **Architecture:** SigLIP2 + PhoBERT + CLIP + ChunkFormer -> CLIPGateFusionV5 -> Linear Head (output size = 1).
- **Loss Function:** `BCEWithLogitsLoss`. 
  - *Xử lý Imbalance:* Dataset có tỷ lệ normal:harmful ~ 1:2.9. Khai báo `pos_weight = số mẫu normal / số mẫu harmful` (~1620/4665 ~ 0.347) hoặc ngược lại tuỳ định nghĩa class positive. Nếu coi `harmful` (1) là positive thì `pos_weight = 1620 / 4665`. Điều này giúp phạt nặng sai số trên class `normal`.
- **Checkpoints:** Lưu model có `Val Binary F1` hoặc `Val ROC-AUC` tốt nhất.

### Stage 2: Phân loại đa nhãn (Multi-label Classification)
- **Mục tiêu:** Phân loại 7 nhãn độc hại chi tiết.
- **Dataset / DataLoader:** Load từ `stage2_train.csv` (tập này chỉ chứa 4,665 samples harmful).
- **Architecture (Warm-start):**
  - Load checkpoint tốt nhất từ Stage 1.
  - Loại bỏ Linear Head 1-neuron của Stage 1. Khởi tạo Linear Head mới với `output size = 7`.
  - *Freeze strategy:* Khuyến nghị freeze các modality backbone (SigLIP2, PhoBERT) trong 1-2 epochs đầu để Head mới hội tụ dần, sau đó unfreeze tất cả với learning rate nhỏ hơn (differential learning rates).
- **Loss Function:** `BCEWithLogitsLoss` cho multi-label.
  - *Xử lý Imbalance (pos_weight):* Tính `pos_weight` cho từng nhãn trong 7 nhãn trên tập Train Stage 2.
    - `pos_weight_i = (N_total - N_i) / N_i` 
    - Ví dụ `information_harm` có 269 mẫu trên 4665 mẫu -> pos_weight ~ (4665-269)/269 = 16.3.
  - *Phương án dự phòng:* Nếu `pos_weight` vẫn khiến mô hình dự đoán nhãn `information_harm` kém, chuyển sang dùng **Asymmetric Loss (ASL)** hoặc **Focal Loss** để tập trung vào các hard positive samples.
- **Checkpoints:** Lưu model theo F1-Macro trung bình của 7 nhãn trên tập Validation.

## 3. Inference Pipeline (Cascade Mapping)

Ở bước test/inference thực tế trên luồng video:

1. **Chạy qua Stage 1 Model:**
   - Trích xuất feature qua các backbone và đẩy qua Stage 1.
   - Tính xác suất: `P_harmful = Sigmoid(Output_Stage1)`.
   - Nếu `P_harmful < 0.5`:
     - **Kết luận:** Video bình thường (`normal_content`).
     - Map kết quả: Cả 7 nhãn đều mang giá trị 0. (Kết thúc luồng xử lý video này, tiết kiệm thời gian chạy Stage 2).

2. **Chạy qua Stage 2 Model (Nếu P_harmful >= 0.5):**
   - Đẩy tiếp các feature (hoặc pass thẳng input) qua Stage 2 Model.
   - Tính toán xác suất 7 nhãn: `P_label_i = Sigmoid(Output_Stage2_i)`.
   - **Đánh giá đa nhãn:** Với từng nhãn `i`, nếu `P_label_i > threshold_i` (ví dụ: 0.5) thì nhãn `i = 1`.
   - **Kết luận:** Trả về danh sách các nhãn có giá trị 1.
   
3. **Logic Fallback (Cân nhắc):**
   - Nếu Stage 1 báo là `harmful` nhưng ở Stage 2 tất cả 7 nhãn đều `< threshold_i`, hệ thống sẽ cần một cơ chế fallback. Gợi ý: Bắt buộc chọn nhãn có xác suất cao nhất `argmax(P_label_i)` để đảm bảo đã là harmful thì phải có ít nhất 1 lỗi vi phạm.

---
Vui lòng xem xét các **Open Questions** và cho tôi biết phản hồi của bạn để tôi có thể tiến hành viết code (các scripts và snippets) cụ thể cho các cấu phần trên.
