"""Read the exact frozen runtime on a CPU-only Modal sandbox.

This reconstructs the pinned baseline image; it is explicitly not an inventory
recorded during the original GPU run. No environment dump or weights are read.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "runs/cloud-20261115T185654Z"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extra-source", action="append", default=[])
    args = parser.parse_args()
    receipt = json.loads((BASELINE / "cloud-evidence.json").read_text())
    frozen = BASELINE / "executed-source"
    provenance = json.loads((BASELINE / "source-provenance.json").read_text())
    for name in ("modal_run.py", "b300_runtime.py", "runtime_compat.py"):
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == provenance["files"][name]
    spec = importlib.util.spec_from_file_location("frozen_context_modal_run", frozen / "modal_run.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    os.environ.setdefault("MODAL_CONFIG_PATH", str(ROOT.parent / "repo/.local-runs/modal-client.toml"))
    os.environ.setdefault("MODAL_SERVER_URL", "https://api.modal.com")
    import modal

    image = launcher.base_image(receipt["image"], True, True, frozen)
    code = r'''
import datetime, hashlib, importlib.metadata, importlib.util, json, os, pathlib, platform, subprocess, sys
site = pathlib.Path(importlib.util.find_spec('vllm').origin).parent.parent
sources = {}
patterns = ['vllm/v1/worker/*.py', 'vllm/v1/worker/gpu/*.py', 'vllm/v1/worker/gpu/model_states/**/*.py', 'vllm/v1/executor/*.py',
            'vllm/v1/core/sched/*.py', 'vllm/v1/attention/backends/*.py',
            'vllm/v1/attention/*.py', 'vllm/models/inkling/nvidia/sconv_swa_attn.py',
            'vllm/forward_context.py', 'vllm/config/compilation.py',
            'vllm/v1/kv_cache_interface.py', 'vllm/v1/core/kv_cache_utils.py',
            'vllm/v1/engine/core.py', 'vllm/engine/arg_utils.py',
            'vllm/model_executor/model_loader/*.py', 'vllm/distributed/parallel_state.py']
for pattern in patterns:
    for path in site.glob(pattern):
        sources[str(path.relative_to(site))] = path.read_text()
for relative in EXTRA:
    path = (site/relative).resolve()
    if not path.is_relative_to(site) or path.suffix not in ('.py','.h','.cuh','.cu'):
        raise ValueError('Only installed source paths accepted')
    sources[relative] = path.read_text()
def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {'argv':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'argv':args,'unavailable':type(e).__name__}
import torch, triton
allow = ['CUDA_MODULE_LOADING','CUDA_DEVICE_MAX_CONNECTIONS','CUDA_VISIBLE_DEVICES',
         'NCCL_CUMEM_ENABLE','NCCL_DEBUG','NCCL_P2P_DISABLE','NCCL_SHM_DISABLE',
         'TORCHINDUCTOR_CACHE_DIR','TORCH_COMPILE_DISABLE','PYTORCH_CUDA_ALLOC_CONF',
         'VLLM_USE_V1','VLLM_ATTENTION_BACKEND','VLLM_ENABLE_V1_MULTIPROCESSING',
         'VLLM_USE_FLASHINFER_SAMPLER','VLLM_NVTX_SCOPES_FOR_PROFILING',
         'TRITON_CACHE_DIR','HF_HOME','CUDA_HOME','CUDA_PATH']
inventory = {'kind':'posthoc-reconstructed-runtime-inventory','schema_version':1,
             'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'scope':'CPU-only reconstruction of pinned baseline image, not original GPU-run environment',
             'gpu_requested':False,'python':sys.version,'platform':platform.platform(),
             'os_release':pathlib.Path('/etc/os-release').read_text(),
             'distributions':sorted([{'name':d.metadata['Name'],'version':d.version} for d in importlib.metadata.distributions()],key=lambda x:x['name'].lower()),
             'active_runtime':{'torch':torch.__version__,'torch_cuda':torch.version.cuda,'triton':triton.__version__,'triton_module':triton.__file__},
             'torch_build_config':torch.__config__.show(),
             'selected_environment':{key:os.environ[key] for key in allow if key in os.environ},
             'commands':[command(['nvcc','--version']),command(['nsys','--version']),command(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader']),command(['uv','--version'])]}
print(json.dumps({'sources':sources,'inventory':inventory}))
'''.replace("EXTRA", repr(args.extra_source))
    app = modal.App("inkling-context-runtime-source-cpu")
    with modal.enable_output(), app.run():
        sandbox = modal.Sandbox.create(
            "uv", "run", "--no-project", "--no-managed-python", "python", "-c", code,
            image=image, app=app, cpu=2, memory=8192, timeout=180)
        try:
            sandbox.wait()
            if sandbox.returncode:
                raise RuntimeError(sandbox.stderr.read())
            result = json.loads(sandbox.stdout.read())
            destination = ROOT / "planning/context-runtime-source"
            destination.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for relative, source in result["sources"].items():
                path = destination / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(source)
                hashes[relative] = hashlib.sha256(source.encode()).hexdigest()
            inventory = result["inventory"]
            inventory["provenance"] = {"image":receipt["image"],"revision":receipt["revision"],
                                       "model":receipt["model"],"baseline_run":BASELINE.name,
                                       "app_id":app.app_id,"resolved_modal_image_id":image.object_id,
                                       "frozen_source_hashes":provenance["files"]}
            inventory["installed_source_hashes"] = hashes
            (ROOT / "planning/runtime-inventory-posthoc.json").write_text(json.dumps(inventory,indent=2)+"\n")
            print(json.dumps({"source_files":len(hashes),"destination":str(destination),"gpu_requested":False}))
        finally:
            sandbox.terminate()


if __name__ == "__main__":
    main()
