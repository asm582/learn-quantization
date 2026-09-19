"""
Step 2: Load a real small model in full precision (fp16) and establish a baseline.

Model: Qwen2.5-0.5B-Instruct (~0.5 billion parameters)
This is small enough to be a good teaching example but real enough to produce
actual coherent text, so we can compare quality before/after quantization.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def print_gpu_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{label}] GPU memory allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return total_bytes / 1e6


if __name__ == "__main__":
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # fp16 (half precision) is our "baseline" -- already 2x smaller than fp32,
    # and standard practice for GPU inference even before any quantization.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()

    n_params = count_params(model)
    print(f"\nParameter count: {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"Model size in memory (weights only): {model_size_mb(model):.1f} MB")
    print_gpu_memory("after loading fp16 model")

    # Run one generation as our quality/speed baseline.
    messages = [{"role": "user", "content": "Explain what quantization means in machine learning, in two sentences."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nGenerated in {elapsed:.2f}s ({80 / elapsed:.1f} tokens/sec)")
    print(f"Output:\n{text}")

    print_gpu_memory("after generation")
