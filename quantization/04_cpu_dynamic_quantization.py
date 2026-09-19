"""
Step 4: PyTorch's built-in dynamic quantization (CPU).

This is a real, standard library feature (torch.quantization.quantize_dynamic)
that works on any CPU, no special GPU kernels needed. It's a good technique
to know because it's the easiest way to shrink a model for CPU deployment.

How "dynamic" quantization works:
  - Weights are quantized to int8 ONCE, ahead of time, and stored that way.
  - Activations (the intermediate outputs as data flows through the network)
    are quantized to int8 ON THE FLY during each forward pass, because their
    range depends on the actual input and can't be known in advance.
  - This is different from "static" quantization, which pre-calibrates
    activation ranges using sample data, and different from what we'll do in
    05 (weight-only quantization with float compute).
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def model_size_mb(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    # quantized layers store weights as int8 buffers, not "parameters" --
    # this helper below covers both cases.
    return total_bytes / 1e6


def real_model_size_mb(model):
    """Save to disk and measure -- the honest way to check quantized model size,
    since quantized weights live in packed buffers that don't show up the same
    way as regular nn.Parameter tensors."""
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading fp32 model on CPU...")
    model_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model_fp32.eval()
    print(f"fp32 model size on disk (state_dict): {real_model_size_mb(model_fp32):.1f} MB")

    print("\nApplying dynamic quantization to all nn.Linear layers...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    print(f"int8 model size on disk (state_dict): {real_model_size_mb(model_int8):.1f} MB")

    messages = [{"role": "user", "content": "Explain what quantization means in machine learning, in two sentences."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    for label, model in [("fp32", model_fp32), ("int8 (dynamic)", model_int8)]:
        start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        elapsed = time.time() - start
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n--- {label} ---")
        print(f"Time: {elapsed:.2f}s ({60 / elapsed:.1f} tok/s on CPU)")
        print(f"Output: {text}")
