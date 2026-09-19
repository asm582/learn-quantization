#!/usr/bin/env bash
# Converts a Hugging Face model to GGUF and quantizes it yourself, using
# llama.cpp's own tools -- the same tools that produce the pre-made GGUF
# files on the Hub, just run by you instead of taken on faith.
#
# Usage: ./convert_and_quantize.sh [HF_MODEL_ID] [QUANT_TYPE]
# Example: ./convert_and_quantize.sh Qwen/Qwen2.5-0.5B-Instruct Q4_K_M

set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
QUANT_TYPE="${2:-Q4_K_M}"
WORKDIR="$(mktemp -d)"
OUT_NAME="$(echo "$MODEL_ID" | tr '/' '_')-${QUANT_TYPE}.gguf"

log() { echo -e "\n\033[1;34m==>\033[0m $*"; }

log "Setting up a venv with the conversion script's dependencies"
python3 -m venv "$WORKDIR/venv"
source "$WORKDIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch transformers gguf sentencepiece

log "Fetching convert_hf_to_gguf.py from llama.cpp (sparse clone, no full repo)"
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/ggml-org/llama.cpp.git "$WORKDIR/llama.cpp" >/dev/null 2>&1
(cd "$WORKDIR/llama.cpp" && git sparse-checkout set gguf-py conversion >/dev/null)

log "Downloading $MODEL_ID (fp16) from Hugging Face"
python3 - "$MODEL_ID" <<'PYEOF'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = sys.argv[1]
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
PYEOF

SNAP_DIR=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL_ID'))
")

log "Converting to F16 GGUF"
python3 "$WORKDIR/llama.cpp/convert_hf_to_gguf.py" "$SNAP_DIR" \
  --outtype f16 --outfile "$WORKDIR/model-f16.gguf"

log "Quantizing to $QUANT_TYPE using llama.cpp's own quantizer"
docker run --rm -v "$WORKDIR":/data --entrypoint /app/llama \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  quantize /data/model-f16.gguf "/data/$OUT_NAME" "$QUANT_TYPE"

mv "$WORKDIR/$OUT_NAME" ./
rm -rf "$WORKDIR"

log "Done: ./$OUT_NAME"
echo "
To serve it in the cluster from setup.sh:
  kubectl cp ./$OUT_NAME \$(kubectl get pod -l app=llama-server -o jsonpath='{.items[0].metadata.name}'):/root/.cache/llama.cpp/custom/$OUT_NAME
Then edit llama-server.yaml's args to use:
  -m /root/.cache/llama.cpp/custom/$OUT_NAME
and re-apply (the Deployment uses 'strategy: Recreate' so this is safe with a single GPU)."
