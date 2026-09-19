# Learn Quantization: From Math to a GPU Cluster on a Laptop

This repo is a hands-on walkthrough of LLM quantization, built by deliberately
using **old, constrained hardware** instead of a modern datacenter GPU. That
constraint turned out to be the most useful part of the exercise: it forces
you to hit the real limits that quantization tooling has, instead of only
seeing the happy path.

The journey goes: quantization math from scratch → a real model → an
industry-standard library that turns out *not* to work on this GPU → two
quantizers we write ourselves → llama.cpp's actual production quantizer →
the whole thing served through a chat UI on Kubernetes.

## Environment

Knowing the exact hardware and versions matters here more than usual, because
several steps below only make sense in light of *how old the GPU is*.

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce GTX 960M, 4GB VRAM, **Maxwell architecture, compute capability 5.0** (~2015) |
| OS | Ubuntu 26.04.1 LTS |
| NVIDIA driver | 580.178.04 (CUDA 13.0) |
| CUDA toolkit (nvcc) | 12.4 |
| Python | 3.12 |
| PyTorch | 2.6.0+cu124 |
| transformers | 5.17.0 |
| bitsandbytes | 0.50.2 |
| Docker | 29.8.0 |
| nvidia-container-toolkit | 1.20.1 |
| kind | v0.27.0 |
| kubectl | v1.37.0 |
| Helm | v3.22.0 |
| llama.cpp | `ghcr.io/ggml-org/llama.cpp:server-cuda` |
| Open WebUI | v0.11.3 |

The model used throughout is **Qwen2.5-0.5B-Instruct** — small enough to load
easily even unquantized on a 4GB card, but a real, modern, instruction-tuned
model that produces genuinely coherent output, so quality differences between
quantization methods are actually visible.

## The journey

### 1. Quantization from scratch, no libraries (`quantization/01_manual_quantization.py`)

Before touching any library, this implements int8 "absmax" quantization by
hand in ~30 lines of numpy: pick a scale from the largest-magnitude value,
divide, round, store as int8; multiply back by the scale to dequantize.

This one script demonstrates the two ideas that everything else builds on:

- **4x memory reduction** (32-bit floats → 8-bit integers), for free, at the
  cost of a small, bounded rounding error.
- **The outlier problem**: a single unusually large value in a tensor forces
  the scale factor to stretch to accommodate it, which quietly wrecks the
  precision of every other value in that tensor. This one problem is *why*
  every serious quantization scheme (LLM.int8(), GPTQ, AWQ, GGUF's k-quants)
  exists in the form it does — they're all, at core, different answers to
  "how do we stop one outlier from ruining everything else."

### 2. A real baseline model (`02_baseline_model.py`)

Loads Qwen2.5-0.5B-Instruct in fp16 on the GPU as a reference point: model
size, memory footprint, and generation speed before any quantization.

### 3. Trying the industry-standard library — and hitting a hardware wall (`03_bnb_8bit.py`)

[bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) is
the standard way most people quantize a model in `transformers`
(`load_in_8bit=True` / `load_in_4bit=True`). Both **fail** on this GPU with a
CUDA error: `named symbol not found`.

The reason: bitsandbytes' fast paths (and vLLM's, and TGI's) are compiled as
custom kernels targeting specific GPU architectures — Turing and newer for
int8, Ampere+ for some int4/Marlin paths. A binary compiled for those
architectures has no matching machine code for a 2015 Maxwell card, so the
CUDA driver has nothing to run. This isn't a bug or a misconfiguration; it's
a hardware requirement most tutorials don't mention because most people
running them have newer GPUs.

### 4. CPU dynamic quantization (`04_cpu_dynamic_quantization.py`)

PyTorch's built-in `torch.quantization.quantize_dynamic` — works on any CPU,
no special hardware needed. Two useful, slightly surprising results:

- Model size dropped **~1.9x, not the full 4x** you'd expect from int8. Dynamic
  quantization only touches `nn.Linear` layers, not the embedding table — and
  this model's 150k-token vocabulary means the embedding matrix is a huge
  fraction of total parameters. Small models with big vocabularies get much
  less benefit from naive per-layer-type quantization schemes.
- Output **quality drifted more than expected** — this is naive, per-tensor
  quantization with no calibration data, unlike GPTQ/AWQ/k-quants which use
  calibration or smarter scaling.

### 5. Writing our own GPU quantizer (`05_gpu_weight_only_int8.py`)

A custom int8 **weight-only** quantizer: store weights as int8 (with a
*per-channel* scale, fixing the single-scale outlier problem from step 1),
dequantize to fp16 on the fly, and run the actual matrix multiply in
ordinary fp16. This is deliberately simple — no custom CUDA kernel, just
generic PyTorch ops — and that's the point: it works on **any** GPU,
including this one, because `cuBLAS`'s plain fp16 matmul has broad hardware
support going back to Kepler.

Two real trade-offs showed up:

- Memory dropped (988MB → 767MB), and output quality stayed close to the
  fp16 baseline.
- **Throughput got worse, not better** (37.8 → 12.3 tok/s). Dequantizing the
  full weight matrix on every forward pass, every single generated token, is
  real, wasted work that a naive implementation pays for and a fused
  kernel wouldn't.

### 6. Proving it's really running on the GPU (`06_verify_gpu.py`)

Two independent checks: confirming the quantized tensors report `cuda:0` as
their device, and sampling `nvidia-smi` utilization in a background thread
while generating — watching it rise to ~98% is a good sanity check that a
"passing" script isn't secretly falling back to CPU.

### 7. Serving it (`07_try_prompt.py`, `08_server.py`)

A one-off prompt script, then a minimal FastAPI server that loads the model
**once** and answers many requests — the difference between a script (pay
model-load cost every run) and a server (pay it once).

## Why llama.cpp (and not vLLM / TGI / bitsandbytes-in-a-server)

Step 3 already showed the pattern: anything relying on custom CUDA kernels
tuned for Turing+ tensor cores (bitsandbytes' fast paths, vLLM's
PagedAttention kernels, TGI's flash-attention backend, Marlin kernels for
GPTQ/AWQ) simply has no compiled code path for a Maxwell GPU. This isn't
specific to bitsandbytes — it's the same wall for the entire class of
"modern GPU serving engine."

[llama.cpp](https://github.com/ggml-org/llama.cpp) takes a different design
stance. Its own maintainers have stated it directly, in the context of a
long-running discussion about Kubernetes support ([ggml-org/llama.cpp#6546](https://github.com/ggml-org/llama.cpp/issues/6546)):

> "I believe llama.cpp spirit is to focus on the on-device/edge deployment."

Practically, that shows up as: broad, portable CUDA kernels (support goes
back to roughly Kepler-era GPUs, plus CPU/Metal/Vulkan backends), and its own
mature quantization format (GGUF, with "k-quant" types like `Q4_K_M`) that's
genuinely more sophisticated than the toy quantizer in step 5 — mixed
bit-widths per tensor (4/5/6/8-bit), block-wise superblock scaling, all
tuned by years of community iteration specifically for exactly this
"consumer/edge hardware" case. It's the right tool for old, small, or
otherwise unusual hardware, which is exactly this laptop's GPU.

## Why Open WebUI

llama.cpp's server (`llama-server`) already exposes an OpenAI-compatible
`/v1/chat/completions` API out of the box. [Open WebUI](https://github.com/open-webui/open-webui)
is a popular, self-hosted chat frontend that speaks that exact API with zero
glue code — point it at the server's URL and you have a full chat UI. No
other reason than: it's the standard, well-maintained option for exactly
this pairing.

## Running it yourself

### One-time host setup (needs root, not automated)

GPU access has to cross two container boundaries here — Docker, then kind's
own nested containerd — so there's real one-time setup:

```bash
# 1. inotify limits (kind's most common failure on any machine, unrelated to GPUs)
sudo sysctl -w fs.inotify.max_user_watches=524288
sudo sysctl -w fs.inotify.max_user_instances=512
echo -e "fs.inotify.max_user_watches=524288\nfs.inotify.max_user_instances=512" | sudo tee /etc/sysctl.d/99-kind.conf

# 2. nvidia-container-toolkit + CDI, so Docker containers can see the GPU at all
#    (install nvidia-container-toolkit for your distro first, then:)
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo systemctl restart docker
```

### Everything else

```bash
cd k8s
./setup.sh
```

This builds NVIDIA's own [`nvkind`](https://github.com/NVIDIA/nvkind) tool,
creates a GPU-enabled `kind` cluster, installs the NVIDIA device plugin, and
deploys `llama-server` + Open WebUI. It checks the one-time setup above and
tells you exactly what to run if anything's missing, rather than failing
halfway through with a confusing error.

```bash
kubectl port-forward svc/open-webui 3000:8080 &
open http://localhost:3000
```

### Quantizing a model yourself

`setup.sh` uses a pre-quantized GGUF from Hugging Face by default. To
reproduce the full pipeline instead — download the original model, convert
it to GGUF, and quantize it with llama.cpp's own tool, all yourself:

```bash
cd k8s
./convert_and_quantize.sh Qwen/Qwen2.5-0.5B-Instruct Q4_K_M
```

## Throughput measured (all on the GTX 960M, 4GB VRAM)

| Setup | Speed | Notes |
|---|---|---|
| fp16 baseline (transformers, GPU) | 37.8 tok/s | starting point |
| fp32 (transformers, CPU) | 9.7 tok/s | |
| int8 dynamic (transformers, CPU) | 16.0 tok/s | only ~1.9x smaller — embeddings untouched |
| our weight-only int8 (transformers, GPU) | 12.3 tok/s | memory ↓22%, but **slower** than fp16 — no fused kernel |
| llama.cpp GGUF Q4_K_M (pre-made) | ~62 tok/s | fused dequant+matmul kernel |
| llama.cpp GGUF Q4_K_M (quantized ourselves) | ~54 tok/s | more fallback tensor types — see below |

The gap between our hand-written quantizer (12.3 tok/s, *slower* than
unquantized) and llama.cpp's (~62 tok/s, much faster) is the single most
important number in this table: **quantizing for memory and quantizing for
speed are different problems**, and the second one needs a real fused
kernel, not just smaller storage.

The size/speed difference between the pre-made GGUF and our own
self-quantized one comes from this model's unusual tensor dimensions (896,
128, etc. — not multiples of 256): llama.cpp's k-quant scheme couldn't apply
pure `Q4_K` to 144 of 290 tensors and fell back to other types, and the exact
fallback choices (and any use of `--imatrix` calibration data) differ
between quantization runs.

## Key lessons

- **Quantization support is hardware-gated.** "Is CUDA available" is not the
  same question as "will this quantization method actually run" — GPU
  generation determines which kernels exist at all.
- **Memory savings and speed are separate axes.** A correct, naive
  quantizer can shrink a model and still make it slower, if the dequant
  step isn't fused into the matmul.
- **Small models pay a disproportionate embedding-table tax.** A 150k-token
  vocabulary can dominate parameter count in a 0.5B model in a way it never
  would in a 70B one — schemes that skip embeddings scale their benefit with
  model size.
- **Container GPU access always needs an explicit device-exposure
  mechanism** (CDI / nvidia-container-runtime), and nested containers (like
  kind's own node-as-a-container design) add a second copy of the same
  problem.
- **Single-GPU Kubernetes clusters need `strategy: Recreate`.** The default
  rolling-update strategy tries to run the old and new pod at once, which is
  impossible when there's exactly one GPU to share.

## How to increase throughput / reduce latency

Roughly in order of impact:

1. **Newer GPU (Turing or later).** Unlocks tensor-core-accelerated int8/int4
   paths (bitsandbytes, vLLM, TensorRT-LLM) and flash-attention. This is the
   single biggest lever available, and the one this whole repo works around
   not having.
2. **Pick the quant level for your actual bottleneck.** `Q4_0`/`Q4_K_S` for
   smallest size and fastest memory-bound decode; `Q5_K_M`/`Q8_0` if quality
   matters more and you have VRAM to spare. Don't assume smaller is always
   faster — more aggressive quantization can mean more fallback/dequant work,
   as seen in this repo's own numbers.
3. **Tune llama.cpp's own flags**: `--n-gpu-layers` (offload as many as fit
   in VRAM), `--ctx-size` (don't allocate more context than you'll use),
   `--batch-size`/`--ubatch-size`, and `--threads` for the CPU-bound parts of
   the pipeline.
4. **Keep the model resident.** The single biggest, easiest win in this repo:
   a one-off script pays the full model-load cost on every run; a server
   (step 7, or the Kubernetes deployment) pays it once. Cache downloaded
   models in a PVC in Kubernetes so pod restarts don't re-download either.
5. **For concurrent users, on capable hardware**: a continuous-batching
   server (vLLM, TGI) will out-throughput a single-request server like
   `llama-server` under load — but only on hardware whose kernels they
   actually support (see the whole first half of this repo).
6. **For latency-critical production**: speculative decoding with a small
   draft model, or a properly fused int8/int4 GPU kernel (Marlin-style)
   instead of the naive dequantize-then-matmul this repo's own quantizer
   uses.
7. **Give slow first-time model loads a generous `startupProbe`** in
   Kubernetes. Liveness probes with tight default thresholds will kill and
   restart a pod that's still downloading a model, creating a crash loop
   that looks like a bug but is really just impatience.

## What's not covered here (natural next steps)

- Rigorous quality evaluation (perplexity, not just eyeballing one response)
  across the different quantization methods above.
- Connecting external tools (e.g. web search) to Open WebUI via MCP — this
  was attempted and partially works (the connection itself wires up fine),
  but reliably getting a small model to *choose* to call the right tool at
  the right time turned out to be its own, separate, unresolved problem.
