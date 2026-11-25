"""Narrow, reversible native-helper dispatch; GPU imports are lazy."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import time

from profiler.artifacts import write_json
from experiments.context_sweep import benchmark as base, runtime

ATTENTION_SHA256 = "ff3164d9d02e1e317698e239d114796a091424f8df7bb1c310381127c7d4b1b1"
HELPER_SHA256 = "f0e68eece41a1844412a49085ef7d69375949facd7a252b140f8800e43414397"
CAPTURE_BYTES_PER_RANK = 512 * 2**20


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_description(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset(), "dtype": str(tensor.dtype),
            "logical_bytes": tensor.numel()*tensor.element_size()}


def copy_strided(tensor, device):
    import torch
    # torch.save retains the entire backing storage. Initialize even the holes
    # of a strided view so no unrelated heap bytes enter a local raw capture.
    span = 1+sum((n-1)*stride for n, stride in zip(tensor.shape, tensor.stride())) if tensor.numel() else 0
    storage = torch.zeros(span, dtype=tensor.dtype, device=device)
    result = storage.as_strided(tensor.shape, tensor.stride())
    result.copy_(tensor)
    return result


class HelperPatch:
    def __init__(self, module, candidate, *, out, rank, instrumented):
        self.module = module
        self.original = module._unshear_rel_logits
        self.fallback = module._use_sheared_fa4
        path = Path(inspect.getsourcefile(self.original))
        self.source_path = path
        if digest(path) != ATTENTION_SHA256:
            raise RuntimeError("Installed native attention source differs from reviewed baseline")
        if hashlib.sha256(inspect.getsource(self.original).encode()).hexdigest() != HELPER_SHA256:
            raise RuntimeError("Original helper source hash differs from reviewed baseline")
        if self.fallback() is not False:
            raise RuntimeError("This B300 experiment requires the original non-sheared fallback")
        self.candidate = candidate
        self.candidate_source_sha256 = digest(inspect.getsourcefile(candidate))
        self.out = Path(out)/"captured-inputs"/f"rank-{rank}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.rank, self.instrumented = rank, instrumented
        self.variant = "native"
        self.context = {"phase": "startup", "length": 0, "stage": "preparation", "iteration": 0, "attempt": 0, "block": 0}
        self.capture_inputs = False
        self.captured = {}
        self.capture_bytes = 0
        self.calls = 0
        self.case_keys = set()
        self.wrapper = self.invoke
        module._unshear_rel_logits = self.wrapper

    def receipt(self):
        return {"target": "vllm.models.inkling2.ops.attention._unshear_rel_logits",
                "source_sha256": ATTENTION_SHA256, "original_helper_sha256": HELPER_SHA256,
                "candidate_source_sha256": self.candidate_source_sha256,
                "installed": self.module._unshear_rel_logits is self.wrapper,
                "restored": self.module._unshear_rel_logits is self.original,
                "fallback_function_unchanged": self.module._use_sheared_fa4 is self.fallback,
                "use_sheared_fa4": self.fallback(), "active_variant": self.variant,
                "scope": "Only this helper global is wrapped; model weights, FA4 fallback/attention consumer, routing and quantization are unchanged."}

    def assert_intact(self, *, check_source=False):
        value = self.receipt()
        if not value["installed"] or not value["fallback_function_unchanged"] or value["use_sheared_fa4"] is not False:
            raise RuntimeError("Helper patch or required B300 fallback changed")
        if check_source and (digest(self.source_path) != ATTENTION_SHA256
                or digest(inspect.getsourcefile(self.candidate)) != self.candidate_source_sha256):
            raise RuntimeError("Native/candidate helper source changed during the experiment")

    def restore(self):
        self.assert_intact()
        self.module._unshear_rel_logits = self.original
        return self.receipt()

    def capture(self, sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent):
        import torch
        from experiments.context_sweep.state import tensor_fingerprint
        # Metadata copies are made only during explicit untimed preparation.
        key_data = {"sheared": tensor_description(sheared), "num_tokens": num_tokens,
                    "cache_seqlens": {**tensor_description(cache_seqlens), "values": cache_seqlens.cpu().tolist()},
                    "cu_seqlens_q": {**tensor_description(cu_seqlens_q), "values": cu_seqlens_q.cpu().tolist()},
                    "rel_extent": rel_extent}
        key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        self.case_keys.add(key)
        if key in self.captured:
            return
        tensors = {"sheared": sheared, "cache_seqlens": cache_seqlens, "cu_seqlens_q": cu_seqlens_q}
        storage_bytes = sum((1+sum((n-1)*stride for n, stride in zip(t.shape, t.stride())))*t.element_size()
                            if t.numel() else 0 for t in tensors.values())
        if self.capture_bytes+storage_bytes > CAPTURE_BYTES_PER_RANK:
            raise RuntimeError("Representative helper capture exceeds declared512MiB/rank cap; no silent omission")
        payload = {name: copy_strided(t, "cpu") for name, t in tensors.items()}
        payload.update(num_tokens=num_tokens, rel_extent=rel_extent)
        path = self.out/f"{key}.pt"
        torch.save(payload, path)
        value = {"capture_id": key, "rank": self.rank, "file": str(path.relative_to(self.out.parent.parent)),
                 "file_bytes": path.stat().st_size, "file_sha256": digest(path), "storage_bytes": storage_bytes,
                 "tensors": {name: {**tensor_description(t), "sha256": tensor_fingerprint(payload[name])} for name, t in tensors.items()},
                 "metadata": key_data, "first_context": dict(self.context),
                 "representative_only": True, "raw_artifact_local_only": True}
        self.captured[key] = value
        self.capture_bytes += storage_bytes
        write_json(self.out/"index.json", {"kind": "actual-relative-bias-inputs-v1", "captures": list(self.captured.values()),
                   "capture_bytes": self.capture_bytes, "max_bytes": CAPTURE_BYTES_PER_RANK})

    def invoke(self, sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent):
        self.calls += 1
        if self.capture_inputs:
            self.capture(sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent)
        c = self.context
        label = (f"relative_bias_helper/{self.variant}/{c['phase']}/L{c['length']}/{c['stage']}/"
                 f"iteration-{c['iteration']}/attempt-{c['attempt']}/block-{c['block']}/call-{self.calls}")
        if self.instrumented:
            import torch
            torch.cuda.nvtx.range_push(label)
        try:
            function = self.original if self.variant == "native" else self.candidate
            return function(sheared, num_tokens, cache_seqlens, cu_seqlens_q, rel_extent)
        finally:
            if self.instrumented:
                torch.cuda.nvtx.range_pop()  # CPU launch range, deliberately no layer-level synchronization.


def worker_initialize(worker, out, instrumented):
    from vllm.models.inkling2.ops import attention
    from .kernel import unshear_rel_logits
    from .correctness import run_correctness
    from experiments.context_sweep.resident_session import profiler_attachment_probe
    result = base.worker_initialize(worker)
    probe = profiler_attachment_probe()
    if not instrumented and (probe.get("maps_checked") is not True or probe.get("known_profiler_detected") is not False):
        raise RuntimeError("Unprofiled relative-bias worker has profiler injection or unavailable attachment check")
    result["native_before_patch_evidence"] = result.pop("native_path_evidence", None)
    patch = HelperPatch(attention, unshear_rel_logits, out=out, rank=worker.rank, instrumented=instrumented)
    worker._relative_patch = patch
    correctness = run_correctness(patch.original, candidate=patch.candidate, device=worker.device, graph=True)
    worker._relative_correctness = correctness
    write_json(Path(out)/f"correctness-rank-{worker.rank}.json", correctness)
    result.update(worker_pid=os.getpid(), helper_patch=patch.receipt(), adversarial_correctness=correctness,
                  profiler_probe=probe,
                  native_path_claim="Native model/router/activation verified before the explicitly recorded helper wrapper; fused variant is an experiment modification.")
    return result


def worker_prepare(worker, phase, length, ids, out):
    patch = worker._relative_patch
    patch.assert_intact(); patch.variant = "native"; patch.capture_inputs = False
    patch.case_keys = set()
    result = base.worker_prepare(worker, phase, length, ids, str(Path(out)/"worker-observations"/f"{phase}-L{length}"))
    # Native-only history preparation above. One shared checkpoint/reference is
    # retained for both variants; switching never calls worker_prepare again.
    result["history_preparation_variant"] = "native"
    return result


def worker_iteration(worker, variant, stage, iteration, attempt, block, capture_inputs=False):
    case = worker._context_case
    directory = Path(case["artifact_dir"])
    path = directory/f"rank-{worker.rank}-{variant}-{stage}-block-{block}-attempt-{attempt}-iteration-{iteration}.json"
    value = {"rank": worker.rank, "variant": variant, "stage": stage, "iteration": iteration,
             "attempt": attempt, "block_index": block, "status": "started"}
    write_json(path, value)
    try:
        value.update(_iteration(worker, variant, stage, iteration, attempt, block, capture_inputs))
        value["status"] = "completed"
        return value
    except Exception as exc:
        value.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        worker._relative_patch.capture_inputs = False
        write_json(path, value)


def _iteration(worker, variant, stage, iteration, attempt, block, capture_inputs):
    import torch
    from vllm.distributed import get_tp_group
    from experiments.context_sweep.state import tensor_fingerprint
    if variant not in ("native", "fused"):
        raise ValueError("Unknown helper variant")
    patch = worker._relative_patch
    patch.assert_intact(); patch.variant = variant; patch.calls = 0; patch.capture_inputs = capture_inputs
    case = worker._context_case
    patch.context = {"phase": case["phase"], "length": case["length"], "stage": stage,
                     "iteration": iteration, "attempt": attempt, "block": block}
    state, checkpoint = case["state"], case["checkpoint"]
    checkpoint.restore(state.scalar_metadata(), verify=True)
    torch.cuda.synchronize(worker.device); get_tp_group().barrier()
    before = {"allocated_bytes": torch.cuda.memory_allocated(worker.device), "reserved_bytes": torch.cuda.memory_reserved(worker.device)}
    torch.cuda.reset_peak_memory_stats(worker.device)
    start, end = case["events"]
    monitor = worker._context_monitor
    monitor.context = {**patch.context, "variant": variant}
    event_start = len(monitor.events)
    label = (f"relative_bias/{case['phase']}/L{case['length']}/{variant}/{stage}/"
             f"iteration-{iteration}/attempt-{attempt}/block-{block}")
    torch.cuda.nvtx.range_push(label)
    try:
        start.record(); begin = time.perf_counter()
        output = runtime.direct_forward(worker.model_runner, state)
        end.record(); torch.cuda.synchronize(worker.device)
        elapsed = time.perf_counter()-begin
    finally:
        torch.cuda.nvtx.range_pop()
    memory = {"before": before, "after_allocated_bytes": torch.cuda.memory_allocated(worker.device),
              "after_reserved_bytes": torch.cuda.memory_reserved(worker.device),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(worker.device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(worker.device)}
    memory["incremental_peak_allocated_bytes"] = memory["peak_allocated_bytes"]-before["allocated_bytes"]
    memory["incremental_peak_reserved_bytes"] = memory["peak_reserved_bytes"]-before["reserved_bytes"]
    event_ms = start.elapsed_time(end)
    compilation = monitor.events[event_start:]
    patch.capture_inputs = False
    monitor.context = {"stage": "validation"}
    get_tp_group().barrier()
    checkpoint.verify_metadata(state.scalar_metadata()); checkpoint.verify_write_boundaries()
    present = output is not None
    if case["output_presence"] is None:
        if variant != "native":
            raise RuntimeError("Only native execution may establish output-rank/reference policy")
        case["output_presence"] = present
    correctness = {"passed": case["output_presence"] == present, "logits_present": present,
                   "state_metadata_fixed": True, "write_boundaries_verified": True, "reference_variant": "native",
                   "reference_established": False, "comparison": "byte_exact"}
    if present:
        cpu = output.detach().cpu().contiguous()
        correctness.update(finite=bool(torch.isfinite(cpu).all()), sha256=tensor_fingerprint(cpu))
        if case["reference"] is None:
            if variant != "native":
                raise RuntimeError("Fused output cannot establish the native reference")
            case["reference"] = cpu.clone(); correctness["reference_established"] = True
        correctness["reference_sha256"] = tensor_fingerprint(case["reference"])
        correctness["byte_exact"] = correctness["sha256"] == correctness["reference_sha256"]
        correctness["passed"] &= correctness["finite"] and correctness["byte_exact"]
    correctness["helper_applicable"] = patch.calls > 0
    correctness["passed"] &= patch.calls > 0 or case["phase"] == "decode"
    return {"wall_seconds": elapsed, "cuda_event_ms": event_ms, "memory": memory, "correctness": correctness,
            "helper_calls": patch.calls, "compilation_events": compilation, "capture_active": case["capture_active"],
            "nvtx_range": label, "worker_pid": os.getpid(), "helper_patch": patch.receipt()}


def _load_capture(patch, key, device):
    import torch
    payload = torch.load(patch.out/f"{key}.pt", map_location="cpu", weights_only=True)
    return {name: copy_strided(value, device) if torch.is_tensor(value) else value for name, value in payload.items()}


def worker_helper_validation(worker, out):
    import torch
    from .correctness import assert_exact
    patch = worker._relative_patch
    if not patch.case_keys:
        if worker._context_case["phase"] != "decode":
            raise RuntimeError("No actual helper input was captured from native prefill")
        return {"rank": worker.rank, "captures": [], "checks": [], "passed": None, "applicable": False,
                "status": "not_applicable", "reason": "Observed zero helper calls: native single-token decode is an unaffected control"}
    reports = []
    for key in sorted(patch.case_keys):
        values = _load_capture(patch, key, worker.device)
        reports.append({"capture_id": key, "result": assert_exact(patch.original, candidate=patch.candidate,
                        **values, label="actual-model-capture-"+key)})
        del values
    torch.cuda.synchronize(worker.device)
    value = {"rank": worker.rank, "captures": [patch.captured[k] for k in sorted(patch.case_keys)],
             "checks": reports, "applicable": True, "passed": all(r["result"].get("passed") is True for r in reports)}
    case = worker._context_case
    write_json(Path(out)/f"helper-correctness-{case['phase']}-L{case['length']}-rank-{worker.rank}.json", value)
    return value


def worker_helper_microbench(worker, out, repeats=20):
    """Separate eager helper study; API intervals include host-dispatch gaps."""
    import torch
    patch = worker._relative_patch
    reports = []
    for key in sorted(patch.case_keys):
        args = _load_capture(patch, key, worker.device)
        functions = {"native": patch.original, "fused": patch.candidate}
        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for event in events:
            event.record()
        warmup_rounds = []
        for _round in range(3):
            warmups = []
            for iteration in range(5):
                for variant, function in functions.items():
                    marker = len(worker._context_monitor.events)
                    output = function(**args)
                    del output
                    torch.cuda.synchronize(worker.device)
                    warmups.append({"iteration": iteration, "variant": variant,
                                    "compilation_events": worker._context_monitor.events[marker:]})
            warmup_rounds.append(warmups)
            if not any(r["compilation_events"] for r in warmups):
                break
        clean_warmups = not any(r["compilation_events"] for r in warmup_rounds[-1])
        rows = []
        for iteration in range(10):
            for variant in (("native", "fused") if iteration % 2 == 0 else ("fused", "native")):
                torch.cuda.synchronize(worker.device)
                before = torch.cuda.memory_allocated(worker.device)
                reserved = torch.cuda.memory_reserved(worker.device)
                torch.cuda.reset_peak_memory_stats(worker.device)
                marker = len(worker._context_monitor.events)
                events[0].record(); begin = time.perf_counter()
                for _ in range(repeats):
                    output = functions[variant](**args)
                    del output
                events[1].record(); torch.cuda.synchronize(worker.device)
                wall = time.perf_counter()-begin
                rows.append({"variant": variant, "iteration": iteration, "repeats": repeats,
                    "wall_us_per_call": wall*1e6/repeats, "cuda_event_us_per_call": events[0].elapsed_time(events[1])*1000/repeats,
                    "before_allocated_bytes": before, "before_reserved_bytes": reserved,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(worker.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(worker.device),
                    "incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated(worker.device)-before,
                    "compilation_events": worker._context_monitor.events[marker:]})
        reports.append({"capture_id": key, "warmup_rounds": warmup_rounds, "observations": rows,
                        "summary": {variant: {"wall_us_per_call": base.stats([r["wall_us_per_call"] for r in rows if r["variant"] == variant]),
                                              "cuda_event_us_per_call": base.stats([r["cuda_event_us_per_call"] for r in rows if r["variant"] == variant])}
                                    for variant in functions},
                        "eligible_for_helper_timing": not patch.instrumented and clean_warmups and not any(r["compilation_events"] for r in rows)})
        del args
    value = {"kind": "relative-bias-helper-microbench-v1", "rank": worker.rank, "instrumented": patch.instrumented,
             "applicable": bool(reports), "status": "observed" if reports else "not_applicable_zero_helper_calls",
             "execution_mode": "eager", "scope": "Helper only; distinct from full-model latency. CUDA event intervals include host launch gaps.",
             "reports": reports}
    case = worker._context_case
    write_json(Path(out)/f"helper-microbench-{case['phase']}-L{case['length']}-rank-{worker.rank}.json", value)
    return value


def worker_evidence(worker):
    import torch
    from experiments.context_sweep.resident_session import profiler_attachment_probe
    patch = worker._relative_patch
    patch.assert_intact(check_source=True)
    probe = profiler_attachment_probe()
    if not patch.instrumented and (probe.get("maps_checked") is not True or probe.get("known_profiler_detected") is not False):
        raise RuntimeError("Plain session acquired profiler injection or attachment check became unavailable")
    value = base.worker_evidence(worker)
    value.update(worker_pid=os.getpid(), device_uuid=str(torch.cuda.get_device_properties(worker.device).uuid),
                 helper_patch=patch.receipt(), profiler_probe=probe, helper_source_unchanged=True)
    return value


def worker_restore(worker):
    return {"rank": worker.rank, "worker_pid": os.getpid(), "helper_patch": worker._relative_patch.restore()}
