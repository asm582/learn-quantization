"""
Step 6: Prove the quantized model is actually running on the GPU, not silently
falling back to CPU.

Two independent checks:
  1. Inspect tensor.device on the quantized weight buffers directly.
  2. Sample nvidia-smi GPU utilization in a background thread while we
     generate tokens, so we can see GPU usage rise during the forward passes.
"""

import threading
import time
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from importlib import import_module
step5 = import_module("05_gpu_weight_only_int8")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# --- background GPU utilization sampler -------------------------------------
samples = []
stop_flag = threading.Event()

def sample_gpu():
    while not stop_flag.is_set():
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        util, mem = out.stdout.strip().split(",")
        samples.append((int(util), int(mem)))
        time.sleep(0.2)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model on cuda and quantizing (weight-only int8)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="cuda")
    step5.quantize_model_(model)
    model.eval()

    # --- Check 1: direct device inspection ---
    print("\n=== Check 1: tensor devices ===")
    print(f"model.device (via first param): {next(model.parameters()).device}")

    found_quantized = False
    for name, module in model.named_modules():
        if isinstance(module, step5.Int8WeightOnlyLinear):
            print(f"Layer '{name}':")
            print(f"  weight_int8.device = {module.weight_int8.device}, dtype = {module.weight_int8.dtype}")
            print(f"  scale.device       = {module.scale.device}, dtype = {module.scale.dtype}")
            assert module.weight_int8.is_cuda and module.scale.is_cuda, "Quantized weights are NOT on GPU!"
            found_quantized = True
            break
    assert found_quantized, "No quantized layers found -- quantize_model_ may not have run."
    print("PASSED: quantized weight buffers are on CUDA.")

    # --- Check 2: watch GPU utilization while generating ---
    print("\n=== Check 2: GPU utilization while generating ===")
    monitor_thread = threading.Thread(target=sample_gpu)
    monitor_thread.start()

    messages = [{"role": "user", "content": "Write a short paragraph about the ocean."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    time.sleep(0.5)  # capture a bit of idle baseline first
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=120, do_sample=False)
    torch.cuda.synchronize()
    time.sleep(0.5)

    stop_flag.set()
    monitor_thread.join()

    utils = [u for u, m in samples]
    mems = [m for u, m in samples]
    print(f"Collected {len(samples)} samples over the run.")
    print(f"GPU utilization: min={min(utils)}%, max={max(utils)}%, avg={sum(utils)/len(utils):.1f}%")
    print(f"GPU memory used (MiB): min={min(mems)}, max={max(mems)}")
    print("\nUtilization trace (%):", utils)

    if max(utils) > 0:
        print("\nPASSED: GPU utilization rose above 0% during generation -- compute is happening on the GPU.")
    else:
        print("\nWARNING: GPU utilization never left 0% -- something is off, check device placement.")
