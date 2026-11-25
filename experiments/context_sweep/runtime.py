"""Native pinned vLLM startup and direct-forward boundary for the context sweep.

This is deliberately specific to the vendor image used by cloud-20261115T185654Z.
It is not a replacement engine or a dummy-model benchmark. The controller must
run it inside the frozen baseline image with its B300 compatibility layer.
"""
from __future__ import annotations

from typing import Any, Callable

MODEL = "thinkingmachines/Inkling-2-Small"
REVISION = "8361e7a31ae7e9966a49a91d3994fde618936c2b"
BASE_IMAGE = "vllm/vllm-openai@sha256:a710ed3b5eba4e205fa1904cae57444446ead088bac575423f9b4a392a9a80a5"
BASELINE_RUN = "cloud-20261115T185654Z"
CAPACITY = 4097
BOUNDARY_ID = "native-tp2-eager-embedding-layers-finalnorm-last-token-head-v1"
NATIVE_SOURCE_SHA256 = {
    "model.py": "54572a0e3046fa9b7c1f39f7b190bc30cfbcfc2868fca7349c6d5e5fac40ad99",
    "moe.py": "0ac77ae575832f9b712f6cb77939f32ed51553d0501bd11df200d0f2c03cb4d8",
    "quant_config.py": "4a734fd2d6b2230aea9653e993b27ac32c6e0ff4f5bf0600b8e687e164b4b4e6",
    "ops/moe.py": "6f39c877fa581417538b484d52d4ca23f91cf1e7348e4f7f4bc24d60aa15095c",
    "ops/activation.py": "1e2b33363922572eec4fe65c36d733401e230c66ecbb58192deb619c49a46a54",
}


def native_source_evidence(*, loaded_model=None) -> dict[str, Any]:
    """Hash the installed native files; on workers verify the actual call aliases."""
    import hashlib
    import importlib.util
    import inspect
    from pathlib import Path
    import sys

    root = Path(importlib.util.find_spec("vllm").origin).parent / "models/inkling2"
    files = {name: {"expected_sha256": expected,
                    "actual_sha256": hashlib.sha256((root/name).read_bytes()).hexdigest()}
             for name, expected in NATIVE_SOURCE_SHA256.items()}
    for value in files.values():
        value["matches_pinned_baseline"] = value["actual_sha256"] == value["expected_sha256"]
    replacements = sorted(name for name in sys.modules
                          if name == "experiments.tokenspeed" or name.startswith("experiments.tokenspeed.")
                          or name.startswith("tokenspeed"))
    result = {"installed_sources": files, "loaded_tokenspeed_module_names": replacements,
              "scope": "Actual installed source hashes and loaded Python call aliases; no module contents or environment dump"}
    if loaded_model is not None:
        model_module = sys.modules["vllm.models.inkling2.model"]
        moe_module = sys.modules["vllm.models.inkling2.moe"]
        route_module = sys.modules["vllm.models.inkling2.ops.moe"]
        activation_module = sys.modules["vllm.models.inkling2.ops.activation"]
        checks = {
            "loaded_model_is_native_class": type(loaded_model) is model_module.Inkling2ForConditionalGeneration,
            "router_alias_is_native_function": moe_module.route is route_module.route,
            "activation_alias_is_native_function": moe_module.swiglu_quantize is activation_module.swiglu_quantize,
            "route_function_module": moe_module.route.__module__,
            "route_function_qualname": moe_module.route.__qualname__,
            "route_function_source_file": str(Path(inspect.getsourcefile(moe_module.route)).relative_to(root)),
        }
        result["loaded_call_checks"] = checks
        result["native_path_verified"] = (all(v["matches_pinned_baseline"] for v in files.values())
            and not replacements and checks["loaded_model_is_native_class"]
            and checks["router_alias_is_native_function"] and checks["activation_alias_is_native_function"]
            and checks["route_function_module"] == "vllm.models.inkling2.ops.moe")
    else:
        result["source_hashes_verified"] = all(v["matches_pinned_baseline"] for v in files.values())
        result["native_path_verified"] = None  # CPU source inspection is not model execution.
    return result


def engine_arguments() -> dict[str, Any]:
    """Exact constructor keywords supported by the inspected EngineArgs."""
    return {
        "model": MODEL,
        "revision": REVISION,
        "tokenizer_revision": REVISION,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "distributed_executor_backend": "mp",
        "max_model_len": CAPACITY,
        "max_num_batched_tokens": CAPACITY,
        "max_num_seqs": 1,
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "async_scheduling": False,
        "quantization": "inkling2",
        "hf_overrides": {"quantization_config": None},
        "limit_mm_per_prompt": {"image": 0},
        "gpu_memory_utilization": 0.90,
        "kv_cache_dtype": "auto",
        "dtype": "auto",
        "cpu_offload_gb": 0,
        "speculative_config": None,
        "seed": 0,
        "disable_log_stats": True,
    }


def create_engine():
    """Load native weights and initialize real TP2 worker/cache machinery.

    EngineCore performs the same native Executor initialization, cache-spec
    discovery, cache-layout resolution, memory sizing, cache binding, and warmup
    as serving. Its scheduler is never given a request by this experiment.
    All of this is outside the benchmark and profiler capture windows.
    """
    import subprocess
    from vllm.config.compilation import CUDAGraphMode
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.abstract import Executor

    # Query through NVML's CLI: initializing torch CUDA in this controller would
    # create an unnecessary GPU context before vLLM spawns its two workers.
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=20,
    )
    devices = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(devices) != 2:
        raise RuntimeError("This experiment requires exactly two visible B300 GPUs")
    for name in devices:
        if "B300" not in name:
            raise RuntimeError("GPU substitution is not authorized")
    config = EngineArgs(**engine_arguments()).create_engine_config()
    assert config.parallel_config.tensor_parallel_size == 2
    assert config.parallel_config.pipeline_parallel_size == 1
    assert config.parallel_config.data_parallel_size == 1
    assert config.speculative_config is None
    assert config.model_config.max_model_len == CAPACITY
    assert config.scheduler_config.max_num_batched_tokens == CAPACITY
    assert not config.scheduler_config.enable_chunked_prefill
    assert not config.cache_config.enable_prefix_caching
    assert config.model_config.enforce_eager
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    engine = EngineCore(config, Executor.get_class(config), log_stats=False)
    assert_idle(engine)
    return engine


def assert_idle(engine) -> None:
    """Direct cache reservations are valid only while the scheduler is idle."""
    if engine.scheduler.get_num_unfinished_requests() != 0:
        raise RuntimeError("Context sweep requires an idle native scheduler")
    if engine.scheduler.has_requests():
        raise RuntimeError("Pending scheduler work would invalidate direct cache ownership")


def collective(engine, function: Callable, *args, timeout: float = 1200, **kwargs):
    """Run a module-level callable(worker, *args, **kwargs) on both TP ranks."""
    assert_idle(engine)
    return engine.model_executor.collective_rpc(
        _invoke_with_config, timeout=timeout, args=(function, args, kwargs)
    )


def _invoke_with_config(worker, function: Callable, args: tuple, kwargs: dict):
    # The stock WorkerWrapper binds this context on selected built-in methods;
    # arbitrary callable RPCs are otherwise invoked without the binding.
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(worker.vllm_config):
        return function(worker, *args, **kwargs)


def direct_forward(runner, prepared):
    """Execute the identical native eager inference boundary on either rank.

    `prepared` supplies input_ids, positions, attn_metadata, slot_mapping and an
    all-False is_padding tensor. Construction and cache restoration occur before
    this call. No tokenization, sampling, request scheduling or host-to-device
    input copy is included. Native in-layer planning and cache writes ARE inside.

    All input tokens pass through the embedding, 48 layers, and final norm; only
    the last row passes through the LM head, as in ordinary next-token inference.
    Returns logits on the TP output rank (other ranks may return None).
    """
    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor, set_forward_context

    count = prepared.input_ids.numel()
    # Reference/state checks may use L+1 outside the measured six-case matrix.
    assert 1 <= count <= CAPACITY
    with torch.inference_mode(), set_forward_context(
        prepared.attn_metadata,
        runner.vllm_config,
        num_tokens=count,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        batch_descriptor=BatchDescriptor(num_tokens=count),
        slot_mapping=prepared.slot_mapping,
        is_padding=prepared.is_padding,
    ):
        hidden = runner.model(input_ids=prepared.input_ids, positions=prepared.positions)
        assert isinstance(hidden, torch.Tensor)
        assert hidden.shape[0] == count
        return runner.model.compute_logits(hidden[-1:])


def worker_runtime_summary(worker) -> dict[str, Any]:
    """Small actual-run receipt without arbitrary environment or model weights."""
    import importlib.metadata
    import torch

    runner = worker.model_runner
    properties = torch.cuda.get_device_properties(worker.device)
    config = runner.vllm_config
    native_path = native_source_evidence(loaded_model=runner.model)
    if native_path["native_path_verified"] is not True:
        raise RuntimeError("Loaded model routing path differs from the pinned native baseline")
    return {
        "rank": worker.rank,
        "local_rank": worker.local_rank,
        "device_name": properties.name,
        "device_uuid": str(properties.uuid),
        "device_total_memory": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "vllm": importlib.metadata.version("vllm"),
        "model_class": f"{type(runner.model).__module__}.{type(runner.model).__qualname__}",
        "native_quantization_class": type(config.quant_config).__qualname__,
        "kv_cache_dtype": str(config.cache_config.cache_dtype),
        "model_dtype": str(config.model_config.dtype),
        "execution_mode": "eager",
        "cudagraph_mode": str(config.compilation_config.cudagraph_mode),
        "max_model_len": config.model_config.max_model_len,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "cache_num_blocks": runner.kv_cache_config.num_blocks,
        "cache_group_count": len(runner.kv_cache_config.kv_cache_groups),
        "token_speed_replacement_enabled": bool(native_path["loaded_tokenspeed_module_names"]),
        "native_path_evidence": native_path,
        "boundary_id": BOUNDARY_ID,
    }
