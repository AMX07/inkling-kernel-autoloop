"""Explicit runtime inventory; never dump the process environment or credentials."""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

SAFE_ENVIRONMENT = (
    "CUDA_MODULE_LOADING", "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_VISIBLE_DEVICES",
    "NCCL_CUMEM_ENABLE", "NCCL_DEBUG", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
    "TORCHINDUCTOR_CACHE_DIR", "TORCH_COMPILE_DISABLE", "PYTORCH_CUDA_ALLOC_CONF",
    "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND", "VLLM_ENABLE_V1_MULTIPROCESSING",
    "VLLM_WORKER_MULTIPROC_METHOD", "VLLM_NVTX_SCOPES_FOR_PROFILING",
    "TRITON_CACHE_DIR", "HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "CUDA_HOME", "CUDA_PATH", "OMP_NUM_THREADS",
)


def _command(argv):
    executable = shutil.which(argv[0])
    if not executable:
        return {"argv": argv, "unavailable": True}
    try:
        result = subprocess.run([executable, *argv[1:]], capture_output=True, text=True, timeout=20)
        return {"argv": argv, "executable": executable, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "executable": executable, "timeout": True}


def runtime_inventory(*, phase: str, gpu_requested: bool):
    import torch
    import triton

    patch = Path("/opt/inkling-runtime-patches.json")
    return {
        "kind": "actual-context-sweep-runtime-inventory", "schema_version": 1,
        "phase": phase, "gpu_requested": gpu_requested,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Captured in this experiment allocation; not reconstructed historical evidence",
        "python": sys.version, "platform": platform.platform(),
        "os_release": Path("/etc/os-release").read_text(),
        "distributions": sorted(
            [{"name": d.metadata["Name"], "version": d.version} for d in importlib.metadata.distributions()],
            key=lambda item: item["name"].lower()),
        "active_runtime": {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                           "triton": triton.__version__, "triton_module": triton.__file__},
        "torch_build_config": torch.__config__.show(),
        "torch_cuda_context_initialized_in_inventory_process": torch.cuda.is_initialized(),
        "compatibility_patches": json.loads(patch.read_text()) if patch.is_file() else None,
        "selected_environment": {key: os.environ[key] for key in SAFE_ENVIRONMENT if key in os.environ},
        "commands": [_command(argv) for argv in (
            ["nvcc", "--version"], ["nsys", "--version"], ["uv", "--version"],
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            ["nvidia-smi", "topo", "-m"],
        )],
    }


def cached_weights_receipt(revision: str):
    """Inspect only config/index bytes and shard filesystem metadata, never weights."""
    snapshot = Path("/cache/huggingface/hub/models--thinkingmachines--Inkling-2-Small/snapshots") / revision
    if not snapshot.is_dir():
        raise FileNotFoundError("Pinned cached model revision is absent; downloads are disabled")
    configs = {}
    for name in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
        source = snapshot / name
        payload = source.read_bytes()
        configs[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    shards = []
    for name in sorted(set(index["weight_map"].values())):
        if Path(name).name != name:
            raise ValueError("Unexpected model shard path")
        source = snapshot / name
        stat = source.stat()
        shards.append({"name": name, "bytes": stat.st_size, "cache_blob": source.resolve().name})
    return {"revision": revision, "snapshot": str(snapshot), "metadata_files": configs,
            "shards": shards, "shard_count": len(shards),
            "total_shard_bytes": sum(item["bytes"] for item in shards),
            "verification": "Pinned snapshot plus config/index hashes and every shard's presence/size; weight bytes not rehashed",
            "weights_volume_read_only": True, "network_downloads_disabled": True}
