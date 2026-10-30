# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Interactive demo for the graduation thesis (object hallucination in COCO 2014 image captioning). The research code lives in `../demo/` (see `../demo/CLAUDE.md` and `../demo/HANDOFF.md`). This webapp shows that pipeline as a React Flow graph where each stage is a node the user runs step by step. The user writes in Vietnamese.

`webapp/` is not a repo of its own. `backend/` and `frontend/` are **two separate nested git repos**, so run git commands inside the one you changed.

## Commands

The backend uses relative imports (`from .pipeline import ...`), so it must be started as the package `webapp.backend.main` **from the KLCN root** (`d:\Documents\BaiTap\KiCuiCung\KLCN`), not from `backend/`. Use the system Python 3.10. The default Python 3.11 on PATH does not have torch/ultralytics/transformers.

```powershell
# Backend (from KLCN root). API docs at http://localhost:8000/docs, health at /api/health
"C:\Users\thinh\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -r webapp/backend/requirements.txt
"C:\Users\thinh\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn webapp.backend.main:app --reload --port 8000

# Frontend (from webapp/frontend)
npm run dev     # http://localhost:3000
npm run build
npm run lint
```

There are no tests in either project.

The frontend is Next.js 16 / React 19. `frontend/AGENTS.md` (pulled in by `frontend/CLAUDE.md`) warns that this Next.js version has breaking changes: read the relevant guide in `frontend/node_modules/next/dist/docs/` before writing Next.js code. `next dev` re-adds that block automatically.

## Architecture

### Backend (`backend/`, FastAPI)

- `pipeline/` is a **standalone copy** of the `../demo/` engine (yolo_objects, generate + local_captioner, clipscore, llm + formula_reranker, vocab, schema). It never imports `demo/` and must not modify it. Changes made in `demo/` do not reach the webapp unless they are copied over by hand.
- `pipeline/config.py` resolves default weight paths relative to the KLCN root: YOLO `demo/yolov8n.pt`, caption checkpoint `demo/coco2014_fixloss_best_model.pth`, vocab `dataset/coco2014_karpathy_baseline_vocab.json`. Override them with `YOLO_MODEL_PATH`, `CAPTION_CHECKPOINT`, `VOCAB_FILE`, `CLIP_MODEL`, `PIPELINE_DEVICE`. The default caption checkpoint file is currently **not present** in `demo/`, so set `CAPTION_CHECKPOINT` to a checkpoint that exists. `backend/weights/` (gitignored) holds copies of the restval checkpoint and retrieval files, but no code references it yet.
- `pipeline/local_captioner.py` hardcodes `NUM_LAYERS = 2`. The `demo/` version infers the layer count from the state dict, and the `restval_*` checkpoints have 3 decoder layers, so they will not load here without porting that change.
- `main.py` lifespan builds the heavy singletons once (YOLO, captioner, CLIP scorer, reranker) and stores them in `engines.py`, which `routes.py` reads. This also avoids a circular import. YOLO's real class names are passed to `vocab.set_object_noun_vocab` at startup, and that set decides which caption words count as object nouns for S_belief and hallucination flags.
- `jobs.py` is an in-RAM job store (lost on restart). One upload creates one job. `STEP_DEPENDENCIES` defines the DAG: `caption` and `detect` need only `upload`, `clipscore` needs `caption`, `rerank` needs `caption` + `detect` + `clipscore`, and `final` needs `rerank`. A step called out of order returns HTTP 409.
- `routes.py`: one endpoint per graph node under `/api/jobs/{job_id}/steps/{caption|detect|clipscore|rerank|final}`, plus `GET /steps/rerank/prompt` (the default LLM prompt, which the UI lets the user edit and send back). `models.py` holds the pydantic response shapes. `routes.py` converts the frozen dataclasses from `pipeline/schema.py` into those shapes.
- Rerank (`pipeline/llm.py`): `mode="formula"` runs the deterministic `FormulaReranker` (`S_total = α·S_base + β·S_align + γ·S_belief`, min-max normalized per image). `mode="llm"` calls an OpenAI-compatible endpoint and **silently falls back to the formula** if no key is set or the call fails. `RerankResult.mode` reports which path actually ran. Env vars (`MODEL_API_KEY`, `MODEL_BASE_URL`, `MODEL`, `FORMULA_ALPHA/BETA/GAMMA/DELTA`) load from `backend/pipeline/.env`, then `../demo/.env`. Values already set in the shell take precedence.
- CORS in `main.py` is `allow_origins=["*"]`. The `FRONTEND_ORIGINS` variable in `backend/README.md` is not read by the code, and the README's `demo/model.pth` default is also out of date. Trust `config.py`.

### Frontend (`frontend/`, Next.js App Router + @xyflow/react + zustand + Tailwind 4)

- `app/page.tsx` defines the fixed node/edge layout and `nodeTypes`. Each pipeline stage has a component in `components/nodes/`, and all of them are wrapped in `NodeShell`.
- `lib/store.ts` (zustand) holds all pipeline state and the `runStep` / `runAll` logic. Its `STEP_DEPENDENCIES` is a **hand-maintained mirror** of `backend/jobs.py::STEP_DEPENDENCIES`. If you change one, change the other. `autoRunMode` runs every step after upload. `rerankMode` picks `llm` or `formula`.
- `lib/api.ts` reads the base URL from `NEXT_PUBLIC_API_BASE_URL` (in `.env.local`, default `http://localhost:8000`) and always sends `ngrok-skip-browser-warning`, because the backend is sometimes exposed through ngrok. Rerank requests deliberately have no client timeout because LLM calls can take more than a minute.
- `lib/types.ts` mirrors `backend/models.py` by hand.
