# Jetson Intent Router (uv + Jetson-friendly)

This project is a local intent router + agent dispatcher designed for:  
* low-latency intent classification (ONNX / TensorRT)
* deterministic routing
* OpenAI-compatible APIs
* Jetson Orin-class hardware (GPU-first, DLA optional but mostly pointless)

It deliberately avoids “assistant frameworks” and other forms of architectural self-harm.

⸻

## What this is
* Clause splitter for compound user requests
* Trainable intent classifier (DistilBERT-class)
* ONNX export + TensorRT build path
* Flask service exposing OpenAI-compatible endpoints
* Config-driven intent → agent → prompt routing
* Pluggable agents (OpenAI API, local command stubs, etc.)

## What this is not
* A chat UI
* A full orchestration framework
* A replacement for your LLM
* Magic

This is a router, not a brain.

⸻

## Environment & Dependency Management (uv)

This project uses uv and pyproject.toml.

**Requirements**  
* Python ≥ 3.9
* uv installed (pip install uv or system package)

### Create environment + install deps
```bash
uv sync
```

This will:
* create .venv/
* install dependencies
* honor uv.lock for reproducibility

### Run commands
```bash
uv run python -m service.app
```

Do not use pip install directly unless you enjoy chaos.

⸻

## Configuration Overview

All behavior is config-driven.

### Intents

`config/intents.yaml`:  
```yaml
intents:
  - MediaLibrary
  - Playback
  - SearchNews
  - CommandControl
  - Unknown
```

### Clause splitting

`config/splitter.yaml`

Regex patterns are applied in order to split compound input:

```yaml
patterns:
  - '\band then\b'
  - '\bthen\b'
  - '\bdon''?t forget to\b'
  - '\band\b'
```

This is intentionally rule-based and boring.

### Intent → agent routing

`config/agents.yaml`

Defines:  
* available agents
* intent → agent mapping
* per-intent prompt templates

⸻

## Training the Intent Classifier

Data format

`data/train.jsonl`, `data/val.jsonl`

One JSON object per line:
```json
{"text": "turn off the lights", "label": "CommandControl"}
```

### Train

```bash
uv run python -m training.train \
  --train data/train.jsonl \
  --val data/val.jsonl \
  --config config/train.yaml \
  --out checkpoints/intent_model
```
Artifacts land in checkpoints/ and are not committed.

⸻

### Export to ONNX

```bash
uv run python -m training.export_onnx \
  --model checkpoints/intent_model \
  --out models/intent_classifier.onnx \
  --seq-len 32
```

Optional but recommended:
```bash
uv run python -m training.simplify_onnx \
  --in models/intent_classifier.onnx \
  --out models/intent_classifier_sim.onnx
```
ONNX files may be committed. TensorRT engines should not be.

⸻

## TensorRT Engine Build (Jetson)

On the Jetson device:
```bash
bash scripts/build_trt.sh \
  models/intent_classifier_sim.onnx \
  models/intent_classifier_fp16.trt \
  32
```
Notes:
* FP16 GPU inference
* Batch size = 1
* Fixed sequence length
* GPU backend by default
* DLA is not recommended for transformer NLP

⸻

## Runtime Backend Selection

Controlled via environment variables:
```env
ROUTER_BACKEND=ort   # ort | trt
ROUTER_MODEL_PATH=models/intent_classifier_sim.onnx
```
`ort` → ONNX Runtime (CPU or CUDA)  
`trt` → TensorRT engine (Jetson GPU)  

⸻

## Running the Service
```bash
cp .env.example .env
uv run python -m service.app
```

Endpoints:  
* GET /healthz
* POST /v1/chat/completions (OpenAI-compatible)

### Example request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "router",
    "messages": [
      {"role": "user", "content": "Find Alien and play it on the TV, then turn off the lights"}
    ]
  }'
```

Response includes:
* agent output
* routing metadata
* clause-level intent decisions

⸻

## Project Layout

`agents/` -    # downstream agents (OpenAI, local command stubs)
`router/` -    # clause splitting + routing logic
`runtime/` -   # inference backends (ORT, TensorRT)
`training/` -  # training + export scripts
`service/` -   # Flask OpenAI-compatible API
`config/` -    # intents, splitter, agents
`data/` -      # example datasets + generators
`scripts/` -   # deployment helpers (TRT build)
`models/` -    # ONNX / TRT artifacts (selective)


⸻

## Git Hygiene (Important)
* `.venv/` is ignored
* `uv.lock` is committed
* TensorRT engines are not committed
* Training checkpoints are ignored
* Secrets live in .env and are ignored

If you commit a .trt engine or a virtualenv, future-you will judge you harshly.

⸻

## Final Note

This system is designed to be:
* fast
* deterministic
* boring
* debuggable

If you feel the urge to “just let the LLM decide the intent,” lie down until the feeling passes.
