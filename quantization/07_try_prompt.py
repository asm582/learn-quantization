"""
Step 7: Send a single prompt to the quantized model and print the response.
Quick sanity/demo script -- edit PROMPT below to try your own.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib import import_module

step5 = import_module("05_gpu_weight_only_int8")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France, and what is it known for?"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="cuda")
    step5.quantize_model_(model)
    model.eval()

    text, elapsed = step5.generate(model, tokenizer, PROMPT, max_new_tokens=150)

    print(f"Prompt: {PROMPT}\n")
    print(f"Response:\n{text}\n")
    print(f"({elapsed:.2f}s)")
