# Ý tưởng và quyết định của khóa luận (cập nhật 2026-10-22)

Bài toán: giảm ảo giác vật thể (nhắc vật thể không có trong ảnh) của bộ sinh chú thích ảnh cỡ nhỏ trên MS COCO 2014
(Karpathy split), đo bằng CHAIRs / CHAIRi, giữ chất lượng câu (BLEU, METEOR, ROUGE-L, CIDEr).

## Phương pháp đang dùng trong bài

1. **Mô hình cơ sở** (`notebooks/01_baseline_...`): CLIP ViT-B/16 + Transformer decoder 3 lớp, train trên train + restval.
2. **Bằng chứng vật thể** E(I) = vật thể YOLO-World thấy ∪ vật thể được ít nhất 3/5 câu truy hồi nhắc (5 câu gần nhất
   trong kho caption train, truy hồi bằng CLIP ViT-B/32).
3. **Chặn logits lúc giải mã** (luật δ = 1): trừ 1 vào logits mọi từ chỉ vật thể nằm ngoài E(I). Chặn mềm: vật thể có
   thật mà bằng chứng bỏ sót vẫn sinh được khi xác suất ban đầu đủ cao.
4. **Chọn lại câu** trên 5 ứng viên (diverse beam search): ba điểm 0,2·S_base + 0,4·S_align + 0,4·S_belief, năm điểm
   thêm 1·S_obj + 0,1·S_retr (bằng chứng vật thể và câu truy hồi).

Kết quả test 5.000 ảnh (YOLO-World), CHAIRs / CIDEr: gốc 7,01 / 119,90 → δ = 1 + năm điểm 3,18 / 120,01.

## Hướng đã thử: lớp chặn học được (evgatedrop, `notebooks/02_de_xuat_evgatedrop.ipynb`)

- Đưa chính lớp chặn vào decoder khi huấn luyện. Bảng β (12 ô: số câu nhắc 0–5 × YOLO thấy/không) học bằng CE hiệu
  chỉnh, gradient chỉ vào β; tắt ngẫu nhiên lớp chặn 50% ảnh mỗi bước (gate dropout, giống classifier-free guidance).
- β hội tụ ≈ [−0,95, −0,36, −0,13, 0, 0, 0 | YOLO thấy: 0] — đúng hình dạng (ít câu nhắc → chặn mạnh) nhưng yếu.
- Nhân β với k chọn trên val (kiểu guidance scale của CFG; MARINE cũng chọn độ mạnh guidance bằng thực nghiệm):
  k = 5 trên test cho trước rerank 4,53 / 118,60, năm điểm 3,28 / 119,88.

## Hướng còn mở

- **evgate-hall** (`research/notebooks/coco2014-clip-vit-evgate-hall.ipynb`, chưa chạy): học β bằng CE + λ · unlikelihood
  trên khối xác suất vật thể vắng (theo COCO instances), chỉ chỉnh β từ checkpoint evgatedrop, 2 epoch. Kỳ vọng ít nhất
  ngang luật; vượt được khi β học được hình dạng khác luật đều (vd chặn nhẹ cả vật thể YOLO thấy nhầm).
- Chọn δ theo quy tắc khai báo trước trên val (cần chạy dòng "Gốc" của Bảng 5 trên 1.000 ảnh val).
- HalluLoss đặt lên decoder đã thử ở thế hệ cũ: không giảm ảo giác (CHAIRs 7,65), không lặp lại.

## Tài liệu gần nhất

Xem `papers/README.md`. Gần với bài nhất: RVCD (bộ phát hiện + truy hồi, trừ logits lúc giải mã), MARINE (DETR + RAM++
làm hướng dẫn kiểu classifier-free guidance), TTH (logits phụ từ bộ phân loại zero-shot trộn vào token vật thể),
constrained beam search (ngược chiều: ép chứa từ có bằng chứng), classifier-free guidance (dropout điều kiện khi train).
