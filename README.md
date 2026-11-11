# Image Captioning Research: giảm ảo giác đối tượng trên MS COCO 2014 và Flickr30k

Tiếng Việt | [English](README_EN.md)

Repo nghiên cứu về chú thích ảnh (image captioning). Hướng hiện tại là giảm **ảo giác đối tượng** (câu mô tả nhắc đến vật thể không có
trong ảnh) mà vẫn giữ chất lượng câu, trên một mô hình nhỏ (CLIP ViT-B/16 + bộ giải mã Transformer). Mã nguồn cũng là phần thực nghiệm
của khóa luận tốt nghiệp. Gồm notebook huấn luyện, script đánh giá và ứng dụng demo.

## 1. Bài toán và ý tưởng

Mô hình chú thích ảnh thường nhắc đến vật thể quen thuộc dù không có trong ảnh (ví dụ thấy bàn bếp thì viết có "dining table").
Khóa luận đo ảo giác bằng **CHAIR** (CHAIRs: % câu có ảo giác, CHAIRi: % lần nhắc vật thể sai, trên 80 lớp COCO) và giữ chất lượng câu
bằng BLEU, METEOR, ROUGE-L, CIDEr, SPICE.

Ý tưởng: kiểm soát ảo giác ở **ba giai đoạn** mà không huấn luyện lại bộ phát hiện.

1. **Khi huấn luyện:** Dynamic Logits (trừ logits của vật thể vắng mặt trong ảnh, chỉ khi huấn luyện) và học tương phản đa mẫu dương với attention pooling.
2. **Khi giải mã:** luật chặn logits cố định `z' = z - δ · 1[từ chỉ vật thể ∉ E(I)]`, với `E(I)` là tập vật thể có bằng chứng: YOLO-World phát hiện **hoặc** được ít nhất `r = 3` trong `K = 5` câu truy hồi nhắc tới.
3. **Khi chọn câu:** chọn lại trong tối đa 5 câu ứng viên bằng năm tín hiệu.

## 2. Pipeline

**Huấn luyện** (`notebooks/02`, chạy trên Kaggle)

```mermaid
flowchart LR
    IMG["Ảnh"] --> CLIP["CLIP ViT-B/16<br/>(đóng băng)"]
    CLIP --> DEC["Bộ giải mã Transformer<br/>3 lớp"]
    DEC --> LOG["Logits"]
    DYN["Dynamic Logits<br/>trừ logits vật thể vắng mặt<br/>(chỉ khi huấn luyện)"] --> LOG
    LOG --> CE["Cross-entropy<br/>(làm trơn nhãn 0,1)"]
    DEC -- "trạng thái ẩn" --> POOL["Attention pooling"]
    POOL --> CON["Mất mát tương phản<br/>(2 câu dương, 3 câu âm)"]
    CE --> LOSS["Tổng mất mát"]
    CON --> LOSS
```

**Suy luận** (`eval/` và `notebooks/eval/`)

```mermaid
flowchart TD
    IMG["Ảnh"] --> YOLO["YOLO-World<br/>yolov8l-worldv2, ngưỡng 0,25"]
    IMG --> RET["CLIP ViT-B/32<br/>top-5 câu gần nhất<br/>(kho train + restval)"]
    IMG --> GEN["Bộ sinh câu"]
    YOLO --> EV["Tập bằng chứng E(I)<br/>= YOLO ∪ vật thể có ≥ 3/5 câu nhắc"]
    RET --> EV
    EV --> BLOCK["Luật chặn logits<br/>trừ δ = 3 ở từ chỉ vật thể ∉ E(I)"]
    GEN --> BLOCK
    BLOCK --> BEAM["Beam search (3) + diverse beam search<br/>(10 chùm, 5 nhóm)"]
    BEAM --> CAND["Tối đa 5 câu ứng viên"]
    CAND --> RERANK["Chọn lại câu năm tín hiệu<br/>S = 0,2·S_base + 0,4·S_align + 0,4·S_belief<br/>+ 1·S_obj + 0,1·S_retr"]
    EV -. "S_obj" .-> RERANK
    RET -. "S_retr" .-> RERANK
    RERANK --> OUT["Câu cuối"]
```

Năm tín hiệu chọn lại câu (mỗi tín hiệu chuẩn hóa min-max trong tập ứng viên của một ảnh):

| Tín hiệu | Ý nghĩa |
|---|---|
| `S_base` | độ tự tin của bộ sinh câu (log-xác suất trung bình của token) |
| `S_align` | độ tương thích ảnh - câu (cosine CLIP ViT-B/16, kiểu CLIPScore) |
| `S_belief` | xác suất câu sau khi cập nhật theo kết quả phát hiện đối tượng (belief revision) |
| `S_obj` | trừ số lớp vật thể trong câu nằm ngoài `E(I)` |
| `S_retr` | độ gần câu ứng viên với các câu truy hồi |

Cấu hình đối chứng: **lớp chặn học được** (bảng mức chặn `δ(n, y)` được học khi huấn luyện, nhân hệ số μ = 6 khi suy luận). Mô hình đề xuất cuối dùng luật chặn **cố định**.

## 3. Kết quả (tập kiểm tra Karpathy, một seed)

MS COCO: 5.000 ảnh kiểm tra, CHAIR tính trên 4.967 ảnh có nhãn vật thể. "Trước" là câu đầu của beam search, "5 tín hiệu" là sau khi chọn lại câu. CHs/CHi: CHAIRs/CHAIRi (%, càng thấp càng tốt).

| Tập | Cấu hình | B-4 | METEOR | ROUGE-L | CIDEr | SPICE | CHs | CHi |
|---|---|---|---|---|---|---|---|---|
| MS COCO | Cơ sở, trước | 36,71 | 28,70 | 57,50 | 119,90 | 21,69 | 7,01 | 4,88 |
| MS COCO | Cơ sở + chọn lại câu | 35,82 | 28,51 | 56,52 | 119,87 | 21,77 | 3,10 | 2,20 |
| MS COCO | Đề xuất, trước | 36,88 | 28,55 | 57,61 | 119,67 | 21,59 | 3,46 | 2,49 |
| MS COCO | **Đề xuất, 5 tín hiệu** | 36,27 | 28,50 | 56,87 | **120,49** | 21,84 | **2,44** | **1,76** |
| Flickr30k | Cơ sở, trước | 32,81 | 24,15 | 52,46 | 73,93 | 17,86 | 6,48 | 4,94 |
| Flickr30k | Cơ sở + chọn lại câu | 31,59 | 24,27 | 51,99 | 74,33 | 18,03 | 4,26 | 3,52 |
| Flickr30k | Đề xuất, trước | 30,21 | 23,85 | 51,97 | 68,06 | 17,73 | 3,24 | 2,62 |
| Flickr30k | **Đề xuất, 5 tín hiệu** | 31,09 | 23,85 | 52,01 | 71,84 | 18,11 | 2,74 | 2,33 |

- Trên MS COCO, mô hình đề xuất giảm CHAIRs từ 7,01 xuống 2,44 (-65,2%) và CHAIRi từ 4,88 xuống 1,76, CIDEr cao hơn 0,59 điểm.
- Trên Flickr30k, CHAIR cũng giảm nhưng CIDEr thấp hơn cơ sở 2,09 điểm: ở đó giảm ảo giác đổi lấy một phần chất lượng câu.
- Câu không ngắn đi và không nhắc ít vật thể hơn đáng kể: số vật thể COCO nhắc mỗi câu là 1,37 (đề xuất) so với 1,46 (cơ sở), recall theo vật thể có trong câu tham chiếu 70,0% so với 71,1%.

**Luật chặn cố định so với lớp chặn học được** (MS COCO test):

| Cấu hình | Vị trí | CIDEr | CHs | CHi |
|---|---|---|---|---|
| Cơ sở, không chặn | trước / 5 tín hiệu | 119,90 / 119,87 | 7,01 / 3,10 | 4,88 / 2,20 |
| Lớp chặn học được, μ = 6 | trước / 5 tín hiệu | 118,27 / 119,54 | 4,37 / 3,04 | 3,08 / 2,20 |
| Luật chặn cố định, δ = 3 | trước / 5 tín hiệu | 119,67 / 120,49 | 3,46 / 2,44 | 2,49 / 1,76 |

Hai cấu hình khác nhau cả cách huấn luyện, nên chênh lệch không quy riêng cho cách chặn được.

**Đóng góp của từng tín hiệu chọn lại câu** (MS COCO test, mô hình đề xuất):

| Cấu hình | CIDEr | CHs | CHi |
|---|---|---|---|
| Trước khi chọn lại câu | 119,67 | 3,46 | 2,49 |
| Năm tín hiệu (r = 3) | 120,49 | 2,44 | 1,76 |
| Bỏ `S_obj` | 120,87 | 2,60 | 1,86 |
| Bỏ `S_retr` | 119,34 | 2,46 | 1,78 |
| Năm tín hiệu, r = 1 | 120,68 | 2,48 | 1,78 |

**Chi phí suy luận (mỗi ảnh, RTX 3050 Laptop):** toàn quy trình 1.855 ms, mô hình cơ sở một câu beam search 357 ms. Truy hồi thêm 52 ms, hai tín hiệu dựa trên câu truy hồi thêm 38 ms, YOLO-World 111 ms, bộ chọn lại câu 197 ms.

**Bật/tắt từng thành phần (MS COCO, 5.000 ảnh, một seed).** "Trước"/"sau" là trước/sau khi chọn lại câu bằng bộ năm tín hiệu. CHAIRs, CHAIRi (%) tính trên 4.967 ảnh. Nguồn bằng chứng dùng cho cả luật chặn và tín hiệu `S_obj`, hai hàng không chặn dùng hợp hai nguồn. δ = 3 chọn trên tập kiểm định với nguồn hợp, giữ nguyên cho các nguồn còn lại. Không đo SPICE.

| Cấu hình | CHAIRs trước | CIDEr trước | CHAIRs sau | CHAIRi sau | CIDEr sau | Recall sau |
|---|---|---|---|---|---|---|
| Bộ sinh cơ sở, không chặn | 7,01 | 119,90 | 3,10 | 2,20 | 119,87 | 70,3 |
| Bộ sinh cơ sở, chặn (YOLO-World + truy hồi) | 3,81 | 119,14 | 2,76 | 1,99 | 119,78 | 69,9 |
| Bộ sinh đề xuất, không chặn | 6,68 | 120,79 | 3,06 | 2,17 | 121,26 | 70,4 |
| Bộ sinh đề xuất, chặn (chỉ YOLO-World) | 2,90 | 118,51 | 2,21 | 1,66 | 118,66 | 68,5 |
| Bộ sinh đề xuất, chặn (chỉ truy hồi) | 2,64 | 114,83 | 1,25 | 1,07 | 112,61 | 59,6 |
| **Bộ sinh đề xuất, chặn (YOLO-World + truy hồi)** | **3,46** | **119,67** | **2,44** | **1,76** | **120,49** | **70,0** |

Nhận xét:
- Luật chặn là thành phần giảm CHAIRs nhiều nhất ở bước sinh câu (6,68 xuống 3,46 với bộ sinh đề xuất, 7,01 xuống 3,81 với bộ sinh cơ sở).
- Dynamic Logits và học tương phản chỉ giảm ảo giác ở mức nhỏ (khi không chặn, CHAIRs 6,68 so với 7,01 trước và 3,06 so với 3,10 sau khi chọn lại câu) nhưng cho CIDEr và recall cao hơn. Khi cả hai bật luật chặn, CHAIRs sau chọn lại câu là 2,44 so với 2,76.
- Chỉ dùng câu truy hồi cho CHAIRs thấp nhất nhưng CIDEr giảm 7,88 điểm và recall chỉ 59,6%, tức mô hình nhắc ít vật thể hơn hẳn. Nguồn hợp cho CIDEr và recall cao nhất, nên được chọn.

Giới hạn: mỗi cấu hình chỉ có một lần chạy (một seed); mô hình đề xuất huấn luyện 48 epoch còn mô hình cơ sở 40 epoch; CHAIR chỉ đo 80 lớp COCO, không đo thuộc tính, số lượng, quan hệ; Flickr30k không có nhãn vật thể COCO nên CHAIR ở đó dựa trên nhãn rút từ câu tham chiếu.

## 4. Cấu trúc thư mục

| Đường dẫn | Vai trò |
|---|---|
| `notebooks/01_baseline_clipvit-transformer-coco2014.ipynb` | Huấn luyện mô hình cơ sở (cross-entropy), ra `restval_baseline_best_model.pth` |
| `notebooks/02_train_proposed_dynlogits_contrastive_attnpool.ipynb` | Huấn luyện **mô hình đề xuất** (Dynamic Logits + học tương phản đa mẫu dương + attention pooling), hai giai đoạn (xem ô đầu notebook), dùng checkpoint epoch 48 (`dynlogits_contrastive_attnpool_epoch48_model.pth`, loss kiểm định thấp nhất) |
| `notebooks/03_ablation_learned_gate_evgatedrop.ipynb` | Phương án đối chứng: lớp chặn học được (EvidenceGate) |
| `notebooks/04_flickr30k_train_baseline.ipynb`, `05_flickr30k_train_proposed.ipynb` | Huấn luyện cơ sở và đề xuất trên Flickr30k |
| `notebooks/eval/coco2014-eval-preranking.ipynb` | Đánh giá câu trước khi chọn lại (beam search thường) |
| `notebooks/eval/coco2014-eval-reranking.ipynb` | Đánh giá toàn quy trình: sinh ứng viên có luật chặn, chọn lại câu |
| `eval/build_yoloworld_report.py` | Dựng báo cáo vật thể YOLO-World (bằng chứng) cho tập cần đánh giá |
| `eval/retrieval_logit_bias_experiment.py` | Sinh câu có luật chặn (`--configs 0,δ`), tùy chọn nguồn bằng chứng `--evidence union/yolo/retr` |
| `eval/logit_bias_final_eval.py` | Ghép các mảnh và chấm đủ chỉ số |
| `eval/sobj_threshold_eval.py`, `sobj_threshold_val.py` | Chấm bộ chọn lại câu năm tín hiệu (ngưỡng r, nguồn bằng chứng, SPICE, độ phủ đối tượng) trên test và val |
| `eval/sretr_experiment.py`, `signal_table.py`, `rerank_with_new_objects.py` | Tín hiệu `S_obj`, `S_retr`; bảng ba/năm tín hiệu; chọn lại câu cho cấu hình không chặn |
| `eval/flickr30k_eval.py` | Đánh giá Flickr30k (chặn logits + chọn lại câu) |
| `eval/run_onoff_ablation.sh`, `onoff_postprocess.py` | Thí nghiệm bật/tắt thành phần (xem mục 6) |
| `webapp/` | Ứng dụng demo FastAPI + Next.js chạy từng bước của pipeline; xem `webapp/backend/README.md` |
| `papers/` | Tài liệu tham khảo |

Module dùng chung ở gốc `demo/`: `local_captioner.py`, `generate.py`, `clipscore.py`, `llm.py`, `formula_reranker.py`, `chair_metric.py`, `chair_synonyms.txt`, `evaluate_baseline.py`, `evaluate_reranked_quality.py`.

## 5. Cài đặt và dữ liệu

- Python 3.10, `pip install -r requirements.txt` (ở gốc `demo/`), PyTorch có CUDA, Java (cho METEOR và SPICE).
- Dữ liệu đặt ở `../dataset/` (ngoài repo): `dataset_coco.json` (Karpathy split), `instances_val2014.json`, `val2014/val2014/` (ảnh), `retrieval_cache/` (kho câu và chỉ số láng giềng), từ điển `coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json` (10.031 từ).
- Checkpoint (`*.pth`, khoảng 500 MB mỗi file) đặt ở gốc `demo/`; bản dùng cho demo nằm trong `webapp/backend/weights/`. Huấn luyện chạy trên Kaggle (T4), đánh giá chạy cục bộ.
- Hằng số của notebook 01, 02, 04, 05 đã đặt theo Bảng 3.2 của khóa luận (decoder 3 lớp, batch 64 cho COCO). Các bản gốc ở `research/notebooks/` ghi giá trị khác (96 / 2 lớp); chưa chạy lại trên Kaggle để kiểm chứng.

## 6. Chạy lại đánh giá (từ thư mục `demo/`)

1. **Bằng chứng YOLO-World** cho tập đánh giá:

   ```
   python Graduation_Thesis/eval/build_yoloworld_report.py --report results/seed43_restval_e10_rerank_test.json --output results/seed43_restval_e10_rerank_test_yoloworld.json
   ```

2. **Chọn δ trên 1.000 ảnh val** (lần quét này chạy một lần trước khi chuyển sang E48 và chưa chạy lại; chọn δ cho CHAIRs thấp nhất sau chọn lại câu, với CIDEr không thấp hơn trường hợp không chặn quá 1 điểm):

   ```
   python Graduation_Thesis/eval/retrieval_logit_bias_experiment.py --report results/val1000_yoloworld.json --max-images 1000 --checkpoint restval_dynlogits_contrastive_attnpool_epoch40_model.pth --vocab-file ../dataset/coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json --configs 0,0 0,1 0,2 0,3 0,4 --output results/ksweep_e40_val1000.json
   ```

3. **Sinh câu trên test** với δ = 3, chia 4 mảnh để dừng giữa chừng không mất kết quả (`--save-rows` giữ ứng viên cho bước 4):

   ```
   python Graduation_Thesis/eval/retrieval_logit_bias_experiment.py --report results/seed43_restval_e10_rerank_test_yoloworld.json --start 0 --max-images 1250 --maj 3 --save-rows --checkpoint dynlogits_contrastive_attnpool_epoch48_model.pth --vocab-file ../dataset/coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json --configs 0,3 --output results/e48_g3_test_part0.json
   ```

   Lặp với `--start 1250`, `2500`, `3750`.

4. **Chấm bộ chọn lại câu năm tín hiệu** (có SPICE và độ phủ đối tượng). `--report` là báo cáo 5.000 ảnh đã ghép các mảnh (xem `eval/onoff_postprocess.py`, hàm ghép):

   ```
   python Graduation_Thesis/eval/sobj_threshold_eval.py --report results/e40_g3_yoloworld_merged.json --output results/e40_g3_sobj_threshold_test.json --thresholds 3 --spice
   ```

   Khóa `five_score_3of5` trong file kết quả là cấu hình cuối; `before_rerank` là câu đầu của beam search.

5. **Thí nghiệm bật/tắt** (hai luồng GPU song song, mỗi luồng chạy lần lượt các đoạn 1.250 ảnh, bỏ qua đoạn đã có):

   ```
   bash Graduation_Thesis/eval/run_onoff_ablation.sh A
   bash Graduation_Thesis/eval/run_onoff_ablation.sh B
   python Graduation_Thesis/eval/onoff_postprocess.py --name cs_g3 --evidence union --spice
   ```

   Tên cấu hình và ý nghĩa nằm ở đầu `run_onoff_ablation.sh`. Giữ tối đa 2 tiến trình GPU cùng lúc.

Cấu hình dùng trong khóa luận: YOLO-World ngưỡng 0,25 (tối đa 20 vật thể), `K = 5`, `r = 3`, `δ = 3`, `J = 5` ứng viên, trọng số `w_base/w_align/w_belief/w_obj/w_retr = 0,2/0,4/0,4/1/0,1`, `ε = 10⁻⁶`.
