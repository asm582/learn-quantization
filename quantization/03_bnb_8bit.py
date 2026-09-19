"""
Step 3: Real 8-bit quantization using bitsandbytes' LLM.int8().

This uses transformers' built-in integration: just pass load_in_8bit=True.
Internally bitsandbytes:
  1. Finds "outlier" feature dimensions (see 01_manual_quantization.py --
     this is exactly the outlier problem, solved properly).
  2. Keeps those outlier dimensions in fp16.
  3. Quantizes everything else to int8 with per-row scales.
  4. Does the matmul as int8 x int8 -> int32, then rescales, then adds back
     the fp16 outlier contribution.

Your GPU (GTX 960M, compute capability 5.0 / Maxwell) predates the Turing
architecture that bitsandbytes' fast int8 kernels were designed for, so this
script will tell us empirically whether it works, and how fast.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def print_gpu_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{label}] GPU memory allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    print("Loading model in 8-bit (bitsandbytes LLM.int8())...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quant_config,
            device_map="cuda",
        )
        model.eval()
        print_gpu_memory("after loading 8-bit model")

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

    except Exception as e:
        print(f"\n8-bit loading/inference FAILED: {type(e).__name__}: {e}")
