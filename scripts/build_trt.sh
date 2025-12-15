#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-}"
ENGINE_PATH="${2:-}"
SEQ_LEN="${3:-32}"

if [[ -z "$ONNX_PATH" || -z "$ENGINE_PATH" ]]; then
  echo "Usage: $0 <model.onnx> <out.engine.trt> [seq_len]"
  exit 1
fi

# On Jetson, trtexec is typically here:
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" ]]; then
  echo "trtexec not found. Install TensorRT or set TRTEXEC=/path/to/trtexec"
  exit 1
fi

echo "Building TensorRT engine:"
echo "  ONNX:    $ONNX_PATH"
echo "  ENGINE:  $ENGINE_PATH"
echo "  SEQ_LEN: $SEQ_LEN"

"$TRTEXEC" \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --builderOptimizationLevel=5 \
  --workspace=2048 \
  --minShapes=input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN} \
  --optShapes=input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN} \
  --maxShapes=input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN}
