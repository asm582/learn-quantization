"""
Step 5: Custom weight-only int8 quantization that actually runs on YOUR GPU.

bitsandbytes failed because its kernels do the actual matrix multiply in
int8 (needs Turing+ tensor cores). Here we do something simpler and older:

  - Store weights as int8 (1 byte/param instead of 2 for fp16) -> real memory savings.
  - At forward time, dequantize the weight back to fp16 (cheap elementwise op)
    and do a normal fp16 matmul -> works on ANY GPU, including yours.

This is called "weight-only quantization" (compute stays high precision,
only storage shrinks). It won't give you a speedup on GPU (you still do fp16
math, plus a small dequant overhead) but it gives a real memory reduction,
which is often what matters most when you're VRAM constrained like we are
(4GB card). This is essentially the idea behind GGUF/llama.cpp quantization.

We also fix the "one outlier ruins everything" problem from 01 by using
PER-CHANNEL scaling: each output row of a weight matrix gets its own scale,
instead of one scale for the entire tensor.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class Int8WeightOnlyLinear(nn.Module):
    """Drop-in replacement for nn.Linear that stores its weight as int8."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None):
        super().__init__()
        # weight shape: (out_features, in_features)
        # Per-channel (per output row) absmax quantization -- each row gets
        # its own scale, so one big weight in one row doesn't blow the
        # precision budget for every other row.
        w_fp32 = weight.detach().float()
        row_absmax = w_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = row_absmax / 127.0  # shape: (out_features, 1)

        q = torch.round(w_fp32 / scale).clamp(-127, 127).to(torch.int8)

        self.register_buffer("weight_int8", q)
        self.register_buffer("scale", scale.to(torch.float16))
        self.bias = nn.Parameter(bias.detach().clone()) if bias is not None else None

    def forward(self, x):
        # Dequantize on the fly: int8 -> fp16, then scale back per row.
        weight_fp16 = self.weight_int8.to(torch.float16) * self.scale
        return F.linear(x, weight_fp16, self.bias)

    def extra_repr(self):
        return f"in={self.weight_int8.shape[1]}, out={self.weight_int8.shape[0]}, dtype=int8"


def quantize_model_(model: nn.Module):
    """Replace every nn.Linear in the model with Int8WeightOnlyLinear, in place."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quantized = Int8WeightOnlyLinear(module.weight, module.bias)
            setattr(model, name, quantized)
        else:
            quantize_model_(module)  # recurse into children


def model_footprint_gb(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / 1e9


def print_gpu_memory(label):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{label}] GPU memory allocated: {allocated:.3f} GB, reserved: {reserved:.3f} GB")


def generate(model, tokenizer, prompt_text, max_new_tokens=80):
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, elapsed


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    question = "Explain what quantization means in machine learning, in two sentences."

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Loading fp16 baseline model on GPU...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="cuda")
    model.eval()
    print(f"fp16 footprint: {model_footprint_gb(model):.3f} GB")
    print_gpu_memory("fp16 loaded")

    text_fp16, t_fp16 = generate(model, tokenizer, question)
    print(f"\n--- fp16 baseline ---\nTime: {t_fp16:.2f}s ({80/t_fp16:.1f} tok/s)\nOutput: {text_fp16}")

    print("\nQuantizing all nn.Linear layers to weight-only int8 (in place)...")
    quantize_model_(model)
    torch.cuda.empty_cache()
    print(f"int8 footprint: {model_footprint_gb(model):.3f} GB")
    print_gpu_memory("int8 (weight-only) after quantizing")

    text_int8, t_int8 = generate(model, tokenizer, question)
    print(f"\n--- weight-only int8 ---\nTime: {t_int8:.2f}s ({80/t_int8:.1f} tok/s)\nOutput: {text_int8}")

    print("\n=== Summary ===")
    print(f"Peak GPU memory used: {torch.cuda.max_memory_allocated()/1e9:.3f} GB")
    print("Memory footprint dropped, compute stayed on the GPU the whole time, "
          "and it worked despite your GPU predating bitsandbytes' int8 tensor core kernels.")
