#!/usr/bin/env bash
# Sets up a local, GPU-enabled Kubernetes cluster (via kind) running llama.cpp
# and Open WebUI. Designed for a single-GPU dev machine (tested on an old
# 4GB laptop GPU -- see README for why that matters).
#
# This script only automates the parts that CAN be automated. A few steps
# need root and are left for you to run by hand (the script checks for them
# and tells you exactly what to run if they're missing).

set -euo pipefail

CLUSTER_NAME="${CLUSTER_NAME:-quant-lab}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
fail() { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
log "Checking prerequisites"
# ---------------------------------------------------------------------------

for cmd in docker kubectl kind helm nvidia-smi; do
  command -v "$cmd" >/dev/null 2>&1 || fail "'$cmd' not found. Install it first."
done

nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi failed to run -- is the NVIDIA driver installed?"

# --- inotify limits: kind's most common "works on my machine" failure ---
WATCHES=$(sysctl -n fs.inotify.max_user_watches)
INSTANCES=$(sysctl -n fs.inotify.max_user_instances)
if [ "$WATCHES" -lt 524288 ] || [ "$INSTANCES" -lt 512 ]; then
  fail "inotify limits too low for kind (watches=$WATCHES instances=$INSTANCES). Run:
  sudo sysctl -w fs.inotify.max_user_watches=524288
  sudo sysctl -w fs.inotify.max_user_instances=512
  echo -e 'fs.inotify.max_user_watches=524288\\nfs.inotify.max_user_instances=512' | sudo tee /etc/sysctl.d/99-kind.conf"
fi

# --- nvidia-container-toolkit + CDI: required for GPU passthrough into Docker ---
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  fail "nvidia-container-toolkit not installed. See README 'GPU passthrough setup' section."
fi
if ! docker info 2>/dev/null | grep -q "Default Runtime: nvidia"; then
  fail "Docker's default runtime is not 'nvidia'. Run:
  sudo nvidia-ctk runtime configure --runtime=docker --set-as-default --cdi.enabled
  sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
  sudo systemctl restart docker"
fi
docker run --rm --device nvidia.com/gpu=all ubuntu:22.04 nvidia-smi -L >/dev/null 2>&1 \
  || fail "GPU is not visible inside a plain container -- CDI setup is incomplete."

log "Prerequisites OK"

# ---------------------------------------------------------------------------
log "Building nvkind (NVIDIA's kind+GPU cluster tool)"
# ---------------------------------------------------------------------------
# nvkind wires up the extra mounts and nested containerd config that kind
# needs to expose a GPU to its (containerized) nodes. Built via Docker so you
# don't need Go installed locally.

if [ ! -x "$SCRIPT_DIR/nvkind" ]; then
  docker run --rm -v "$SCRIPT_DIR":/out golang:1.24 bash -c "
    git clone --depth 1 https://github.com/NVIDIA/nvkind.git /src/nvkind &&
    cd /src/nvkind && go build -o /out/nvkind ./cmd/nvkind
  "
fi
chmod +x "$SCRIPT_DIR/nvkind"

# ---------------------------------------------------------------------------
log "Creating GPU-enabled kind cluster: $CLUSTER_NAME"
# ---------------------------------------------------------------------------

if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  echo "Cluster '$CLUSTER_NAME' already exists, skipping creation."
else
  "$SCRIPT_DIR/nvkind" cluster create --name="$CLUSTER_NAME" --wait=120s
fi

CTX="kind-$CLUSTER_NAME"

# ---------------------------------------------------------------------------
log "Installing NVIDIA device plugin"
# ---------------------------------------------------------------------------
# The default chart expects Node Feature Discovery labels we don't have in a
# minimal setup, so: label the node manually, and force the plugin pod onto
# the 'nvidia' RuntimeClass (nvkind creates this) so it can actually reach
# NVML instead of just failing to find any GPU.

WORKER_NODE=$(kubectl --context="$CTX" get nodes -o jsonpath='{.items[?(@.spec.taints==null)].metadata.name}')
kubectl --context="$CTX" label node "$WORKER_NODE" nvidia.com/gpu.present=true --overwrite

helm repo add nvdp https://nvidia.github.io/k8s-device-plugin >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade -i --kube-context="$CTX" --namespace nvidia --create-namespace \
  nvidia-device-plugin nvdp/nvidia-device-plugin \
  --set runtimeClassName=nvidia

echo "Waiting for the device plugin to register nvidia.com/gpu..."
for i in $(seq 1 30); do
  GPU_COUNT=$(kubectl --context="$CTX" get node "$WORKER_NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "")
  [ -n "$GPU_COUNT" ] && break
  sleep 2
done
[ -n "$GPU_COUNT" ] || fail "nvidia.com/gpu never showed up as allocatable. Check: kubectl --context=$CTX logs -n nvidia -l app.kubernetes.io/name=nvidia-device-plugin"
echo "GPU allocatable on $WORKER_NODE: $GPU_COUNT"

# ---------------------------------------------------------------------------
log "Deploying llama.cpp and Open WebUI"
# ---------------------------------------------------------------------------

kubectl --context="$CTX" apply -f "$SCRIPT_DIR/llama-server.yaml"

# A random, persistent secret key -- without this, Open WebUI's connected
# tools break with decryption errors every time the pod restarts.
if ! kubectl --context="$CTX" get secret open-webui-secret >/dev/null 2>&1; then
  kubectl --context="$CTX" create secret generic open-webui-secret \
    --from-literal=WEBUI_SECRET_KEY="$(openssl rand -hex 32)"
fi
kubectl --context="$CTX" apply -f "$SCRIPT_DIR/open-webui.yaml"

log "Waiting for llama-server (first run downloads the model, can take a few minutes)"
kubectl --context="$CTX" rollout status deployment/llama-server --timeout=300s

log "Waiting for Open WebUI (first run downloads an embedding model, can take a few minutes)"
kubectl --context="$CTX" rollout status deployment/open-webui --timeout=300s

# ---------------------------------------------------------------------------
log "Done"
# ---------------------------------------------------------------------------
cat <<EOF

Cluster '$CLUSTER_NAME' is up. To use it:

  kubectl config use-context $CTX

  # Chat UI
  kubectl port-forward svc/open-webui 3000:8080 &
  open http://localhost:3000

  # Raw OpenAI-compatible API
  kubectl port-forward svc/llama-server 8090:8080 &
  curl http://localhost:8090/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}'

Note: 'kubectl port-forward' dies whenever its target pod is replaced (e.g.
after 'kubectl apply' changes a Deployment). If a forwarded port suddenly
stops responding, just re-run the port-forward command.

To tear everything down:
  kind delete cluster --name $CLUSTER_NAME
EOF
