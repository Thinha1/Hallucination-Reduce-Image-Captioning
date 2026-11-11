# Backend — Branching Captioning Pipeline API

FastAPI wrapper around a **standalone copy** of the `demo/` captioning engine (see
`pipeline/`), extended with the proposed method: caption retrieval, the fixed logit-blocking
rule and the five-signal reranking (`pipeline/retrieval.py`). `demo/` is never imported; all
model files live in `weights/`.

Each pipeline stage is its own endpoint so the frontend (React Flow) can run the graph one node
at a time instead of all at once.

## Cai dat & chay

Dung Python 3.10 co san torch/ultralytics/transformers/fastapi/uvicorn (kiem tra: Python 3.11
mac dinh tren PATH KHONG co cac package nay):

```powershell
"C:\Users\thinh\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r webapp/backend/requirements.txt
```

Chay server (tu thu muc chua `webapp/`, vi du `d:\Documents\BaiTap\KiCuiCung\KLCN`):

```powershell
"C:\Users\thinh\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn webapp.backend.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

## Mo hinh va du lieu (thu muc `weights/`)

Moi file mo hinh nam trong `webapp/backend/weights/` (khong tro ra `demo/` nua), nen copy ca thu
muc `webapp` di dau cung chay duoc. Thu muc nay nam trong `.gitignore`.

| File | Vai tro |
|---|---|
| `models.json` | Danh sach checkpoint sinh chu thich hien trong o chon mo hinh tren giao dien (id, ten, file, vocab, delta mac dinh) |
| `restval_dynlogits_contrastive_attnpool_epoch40_model.pth` | Mo hinh de xuat (delta mac dinh 3) |
| `restval_baseline_best_model.pth` | Mo hinh co so (delta mac dinh 0, khong chan) |
| `coco2014_karpathy_restval_dynlogits_contrastive_attnpool_vocab.json` | Vocab 10.031 tu, dung chung cho hai checkpoint tren |
| `yolov8l-worldv2.pt` | YOLO-World nhan dien vat the |
| `clip-vit-base-patch16/` | CLIP ViT-B/16 cho CLIPScore va kien truc encoder cua captioner |
| `clip-vit-base-patch32/` | CLIP ViT-B/32 cho truy hoi cau va S_retr |
| `retrieval/train_captions.json`, `retrieval/train_caption_emb_b32.npy` | Kho truy hoi: 566.747 cau train/restval va embedding B/32 (fp16) |

Them checkpoint moi: chep `.pth` + vocab vao `weights/` roi them mot muc vao `models.json`.

Tham so cua quy trinh de xuat nam trong `pipeline/config.py`: K = 5, r = 3, w_obj = 1, w_retr = 0,1,
alpha/beta/gamma = 0,2/0,4/0,4 (khong doc tu `.env`).

## Bien moi truong (tuy chon)

| Bien | Mac dinh | Y nghia |
|---|---|---|
| `WEIGHTS_DIR` | `webapp/backend/weights` | Thu muc chua mo hinh |
| `YOLO_MODEL_PATH` | `weights/yolov8l-worldv2.pt` | Duong dan checkpoint YOLO |
| `PIPELINE_DEVICE` | `cuda` neu co, khong thi `cpu` | Torch device cho moi mo hinh |
| `PRELOAD_ALL_MODELS` | `1` | Nap san tat ca checkpoint trong `models.json` luc khoi dong (dat `0` de chi nap mo hinh mac dinh) |
| `AUTO_LOAD_RETRIEVAL` | `1` | Tu nap kho truy hoi luc khoi dong (dat `0` de nap tay bang nut tren giao dien) |

## Endpoints

| Buoc | Method & Path |
|---|---|
| Danh sach mo hinh | `GET /api/models` |
| Kho truy hoi | `GET /api/retrieval` (trang thai), `POST /api/retrieval/load` (nap kho) |
| 1. Input | `POST /api/jobs` (multipart `image`) |
| 2A. YOLO-World | `POST /api/jobs/{job_id}/steps/detect` |
| 2B. Truy hoi cau | `POST /api/jobs/{job_id}/steps/retrieve` (can nap kho truoc) |
| 3. Sinh chu thich + chan logits | `POST /api/jobs/{job_id}/steps/caption` (body `{"model_id": ..., "delta": ...}`, can 2A + 2B) |
| 4. CLIPScore | `POST /api/jobs/{job_id}/steps/clipscore` |
| 5. Chon lai cau | `POST /api/jobs/{job_id}/steps/rerank` (cong thuc nam tin hieu, khong dung LLM) |
| 6. Final | `GET /api/jobs/{job_id}/steps/final` |
| — | `GET /api/jobs/{job_id}` (snapshot toan bo job) |
| — | `GET /api/jobs/{job_id}/image` (serve anh da upload) |

Goi step sai thu tu tra ve HTTP 409 kem danh sach step con thieu. Chay lai buoc 3 (vi du doi mo
hinh) se dua buoc 4-6 ve "chua chay".

State luu trong RAM (khong DB) — mat khi restart server, du dung cho demo 1 nguoi dung.
