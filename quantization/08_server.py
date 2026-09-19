"""
Step 8: Serve the weight-only int8 quantized model over HTTP.

This is the missing piece between "run a script that loads the model every
time" and "a real server": load the model ONCE at startup, keep it resident
on the GPU, and answer requests as they come in.

This is a toy single-request server (no batching, no streaming, no queueing --
that's what vLLM/TGI add on top). But it's the same basic shape: model in
memory, HTTP in front of it.

Run:
    python 08_server.py

Then in another terminal:
    curl -X POST http://localhost:8000/generate \
      -H "Content-Type: application/json" \
      -d '{"prompt": "What is the capital of India?", "max_new_tokens": 100}'
"""

import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib import import_module

step5 = import_module("05_gpu_weight_only_int8")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

app = FastAPI(title="Quantized Qwen2.5-0.5B server")

# --- Load once, at import/startup time, not per-request ---
print(f"Loading {MODEL_NAME} and quantizing to weight-only int8...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="cuda")
step5.quantize_model_(model)
model.eval()
print("Model ready and resident on GPU.")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100


class GenerateResponse(BaseModel):
    response: str
    time_seconds: float
    tokens_per_second: float


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    text, elapsed = step5.generate(model, tokenizer, req.prompt, max_new_tokens=req.max_new_tokens)
    return GenerateResponse(
        response=text,
        time_seconds=round(elapsed, 2),
        tokens_per_second=round(req.max_new_tokens / elapsed, 1),
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": str(next(model.parameters()).device)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
