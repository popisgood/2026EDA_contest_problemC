#!/usr/bin/env bash
# Download FloorSet Lite v2 training set (~6.6 GB compressed, ~24 GB extracted).
#
# Usage:
#   bash scripts/download_floorset_train.sh
#
# Optional: set HF_TOKEN env var first for faster downloads
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx     # get from https://huggingface.co/settings/tokens
#   bash scripts/download_floorset_train.sh
#
# Resumable: re-running picks up from where it left off (HF cache handles partial files).

set -euo pipefail

# ---- paths ------------------------------------------------------------------
# IMPORTANT: `export` so the Python heredoc below can see these via os.environ.
export DATA_ROOT="${DATA_ROOT:-/home/pop/IntelLabs_Floorset/FloorSet/floorset_train_data}"
export VENV_PATH="${VENV_PATH:-/home/pop/IntelLabs_Floorset/FloorSet/venv}"

mkdir -p "$DATA_ROOT"
echo "[download] target directory: $DATA_ROOT"

# ---- activate venv + ensure huggingface_hub ---------------------------------
if [ ! -d "$VENV_PATH" ]; then
    echo "[download] venv not found at $VENV_PATH; aborting"
    exit 1
fi
source "$VENV_PATH/bin/activate"
pip install --quiet huggingface_hub

# ---- download ---------------------------------------------------------------
# We use snapshot_download with `allow_patterns` so HF gives us the resumable
# parallel transfer rather than the streamed single-file API.
python - <<PYEOF
import os, time
from pathlib import Path
from huggingface_hub import snapshot_download

data_root = Path(os.environ["DATA_ROOT"])
print(f"[hf] downloading LiteTensorData_v2.tar.gz into {data_root}")
start = time.time()
local_path = snapshot_download(
    repo_id="IntelLabs/FloorSet",
    repo_type="dataset",
    allow_patterns=["LiteTensorData_v2.tar.gz"],
    local_dir=str(data_root),
    # max_workers default = 8 = parallel chunks; faster than single-stream
)
dt = time.time() - start
print(f"[hf] done in {dt:.1f}s -> {local_path}")
PYEOF

# ---- extract ----------------------------------------------------------------
TAR_PATH="$DATA_ROOT/LiteTensorData_v2.tar.gz"
if [ ! -f "$TAR_PATH" ]; then
    echo "[extract] tarball missing at $TAR_PATH"
    exit 2
fi

echo "[extract] uncompressing (this will take 5-10 min)..."
tar -xzf "$TAR_PATH" -C "$DATA_ROOT" --checkpoint=10000 --checkpoint-action=dot
echo

# ---- verify -----------------------------------------------------------------
WORKER_DIRS=$(find "$DATA_ROOT/floorset_lite" -maxdepth 1 -type d -name 'worker_*' | wc -l)
LAYOUTS=$(find "$DATA_ROOT/floorset_lite" -name 'layouts*.pt' | wc -l)
SIZE=$(du -sh "$DATA_ROOT/floorset_lite" | cut -f1)
echo "[verify] $WORKER_DIRS worker dirs, $LAYOUTS layouts*.pt files, $SIZE total"

# ---- clean up tarball -------------------------------------------------------
if [ "$WORKER_DIRS" -ge 50 ]; then
    rm -f "$TAR_PATH"
    echo "[clean] removed tarball"
else
    echo "[clean] extraction looks incomplete ($WORKER_DIRS dirs); KEEPING tarball at $TAR_PATH"
    exit 3
fi

echo
echo "==== READY ===="
echo "Training set: $DATA_ROOT"
echo "Total cases : ~$((WORKER_DIRS * LAYOUTS / WORKER_DIRS * 112))"
echo
echo "Next: train with"
echo "  python -m ml.train --data $DATA_ROOT --device cuda --batch 64 --epochs 10 --workers 4"
