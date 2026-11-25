"""Six native TP2 context cases; no GPU work without --run.

Run this module in the frozen vendor image via uv. The default child has no
Nsight attachment; --profile is a separate instrumented child, never a baseline.
CUDA operations occur only in the two native workers, not in the controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
import traceback
from pathlib import Path

from profiler.artifacts import manifest, write_json
from . import runtime

LENGTHS = (256, 1024, 4096)
WARMUPS = 5
MEASUREMENTS = 10
VOCABULARY = 200058
NUMERICAL_POLICY = {
    "same_case_logits": "finite and byte-exact equality to first prepared native forward",
    "state_restore": "byte-exact cache and native metadata restoration before every invocation",
    "writes": "no history/padding modification outside native scheduled write slots",
    "cross_path_reference": "not a completion gate: no validated tolerance for different native prefill/decode kernels",
}


def input_sequence(seed):
    rng = random.Random(seed)
    ids = [rng.randrange(VOCABULARY) for _ in range(max(LENGTHS) + 1)]
    return ids, {
        "generator": "Python random.Random(seed).randrange(200058)", "seed": seed,
        "vocabulary_size": VOCABULARY, "token_ids": ids,
        "sha256": hash_ids(ids),
        "prefix_sha256": {str(n): hash_ids(ids[:n]) for n in (*LENGTHS, *(n+1 for n in LENGTHS))},
    }


def hash_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def stats(values):
    if not values or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
        raise ValueError("Statistics require finite measured observations")
    return {"count": len(values), "mean": statistics.mean(values),
            "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def validate_ranks(rows):
    if not isinstance(rows, list) or len(rows) != 2 or {r.get("rank") for r in rows} != {0, 1}:
        raise ValueError("Exactly one observation from each of TP ranks 0 and 1 is required")
    return sorted(rows, key=lambda r: r["rank"])


def combine_iteration(rows, rpc_seconds):
    rows = validate_ranks(rows)
    wall = [r["wall_seconds"] for r in rows]
    events = [r["cuda_event_ms"] for r in rows]
    stats(wall + events + [rpc_seconds])
    if min(wall + events) <= 0:
        raise ValueError("Nonpositive native timing")
    return {"ranks": rows, "two_rank_completion_seconds": max(wall),
            "controller_rpc_seconds": rpc_seconds,
            "correctness_passed": all(r["correctness"]["passed"] for r in rows)
            and any(r["correctness"]["logits_present"] for r in rows),
            "compilation_observed": any(r["compilation_events"] for r in rows)}


class CompileMonitor:
    """Observe pinned native JIT callbacks plus Triton autotuner benchmark calls.

    No timing heuristic discards outliers. Hooks do no work on ordinary kernel
    launches; unmonitored CUDA/C++ lazy module loads remain an explicit limit.
    """
    def __init__(self):
        import functools
        from vllm.utils import jit_monitor
        from triton.runtime.autotuner import Autotuner

        self.events = []
        self.context = {}
        original = jit_monitor._handle_jit_event

        @functools.wraps(original)
        def hook(**kwargs):
            self.events.append({**self.context, **{k: str(kwargs[k])[:240]
                               for k in ("backend", "event", "fn_name") if k in kwargs}})
            return original(**kwargs)
        jit_monitor._handle_jit_event = hook
        jit_monitor.activate(mode="warn", verbose=False)
        original_bench = Autotuner._bench

        @functools.wraps(original_bench)
        def bench(instance, *args, **kwargs):
            self.events.append({**self.context, "backend": "Triton", "event": "autotuner benchmark",
                                "fn_name": str(getattr(instance, "base_fn", "autotuner"))[:240]})
            return original_bench(instance, *args, **kwargs)
        Autotuner._bench = bench
        self.coverage = {"native_jit_monitor_active": jit_monitor.is_active(),
                         "native_jit_callback": True, "triton_autotuner_bench": True,
                         "limitations": ["No claim to detect every C++/CUDA lazy load or external autotuner.",
                                         "No latency-based selective sample rejection."]}


def worker_initialize(worker):
    import torch
    worker._context_monitor = CompileMonitor()
    worker._context_case = None
    torch.cuda.synchronize(worker.device)
    result = runtime.worker_runtime_summary(worker)
    result["compilation_monitor"] = worker._context_monitor.coverage
    return result


def worker_prepare(worker, phase, length, ids, artifact_dir=None):
    import torch
    from .state import build_native_sequence_state

    worker._context_case = None  # Release previous case's snapshots and CPU logits.
    runner = worker.model_runner
    monitor = worker._context_monitor
    monitor.context = {"phase": phase, "length": length, "stage": "preparation"}
    if phase == "prefill":
        state = build_native_sequence_state(runner, ids[:length], capacity_tokens=length)
        state.zero()
    elif phase == "decode":
        prep = build_native_sequence_state(runner, ids[:length], capacity_tokens=length+1)
        prep.zero()
        runtime.direct_forward(runner, prep)
        torch.cuda.synchronize(worker.device)
        state = build_native_sequence_state(runner, ids[length:length+1],
                                            history_tokens=length, capacity_tokens=length+1)
        del prep
    else:
        raise ValueError("Unknown phase")
    checkpoint = state.make_checkpoint()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # CUDA event creation is lazy; initialize both handles outside every timer.
    start.record(); end.record(); torch.cuda.synchronize(worker.device)
    inner = getattr(runner.model, "model", None)
    sp = getattr(inner, "seq_parallel", None)
    minimum = getattr(sp, "min_tokens", None)
    worker._context_case = {"state": state, "checkpoint": checkpoint, "reference": None,
                            "events": (start, end), "phase": phase, "length": length,
                            "capture_active": False, "output_presence": None,
                            "artifact_dir": artifact_dir}
    return {"rank": worker.rank, "state": state.evidence(), "checkpoint": checkpoint.evidence(),
            "sequence_parallel": {"available": sp is not None, "min_input_tokens": minimum,
                                  "selected": sp is not None and state.query_tokens >= minimum},
            "execution_mode": "eager"}


def worker_capture(worker, enabled):
    import torch
    from vllm.distributed import get_tp_group
    torch.cuda.synchronize(worker.device)
    get_tp_group().barrier()  # Uses native CPU group, outside forward interval.
    fn = torch.cuda.cudart().cudaProfilerStart if enabled else torch.cuda.cudart().cudaProfilerStop
    result = fn()
    torch.cuda.check_error(result)  # Compare the native cudaError enum, not its Python representation.
    worker._context_case["capture_active"] = enabled
    return {"rank": worker.rank, "capture_active": enabled, "return_code": int(result)}


def worker_iteration(worker, stage, iteration, attempt):
    """Persist each rank independently, even if the other rank's RPC fails."""
    directory = worker._context_case.get("artifact_dir")
    path = (Path(directory)/f"rank-{worker.rank}-{stage}-attempt-{attempt}-iteration-{iteration}.json"
            if directory else None)
    observation = {"rank": worker.rank, "stage": stage, "iteration": iteration,
                   "attempt": attempt, "status": "started"}
    if path:
        write_json(path, observation)
    try:
        observation = {"status": "completed", **_worker_iteration(worker, stage, iteration, attempt)}
        return observation
    except Exception as exc:
        observation.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if path:
            write_json(path, observation)


def _worker_iteration(worker, stage, iteration, attempt):
    import torch
    from vllm.distributed import get_tp_group
    from .state import tensor_fingerprint

    case = worker._context_case
    state, checkpoint = case["state"], case["checkpoint"]
    checkpoint.restore(state.scalar_metadata(), verify=True)
    torch.cuda.synchronize(worker.device)
    get_tp_group().barrier()
    start, end = case["events"]
    before = {"allocated_bytes": torch.cuda.memory_allocated(worker.device),
              "reserved_bytes": torch.cuda.memory_reserved(worker.device)}
    torch.cuda.reset_peak_memory_stats(worker.device)
    monitor = worker._context_monitor
    monitor.context = {"phase": case["phase"], "length": case["length"], "stage": stage,
                       "iteration": iteration, "attempt": attempt}
    event_index = len(monitor.events)
    label = f"context/{case['phase']}/L{case['length']}/{stage}/iteration-{iteration}/attempt-{attempt}"
    torch.cuda.nvtx.range_push(label)
    try:
        start.record()
        begin = time.perf_counter()
        output = runtime.direct_forward(worker.model_runner, state)
        end.record()
        torch.cuda.synchronize(worker.device)
        elapsed = time.perf_counter() - begin
    finally:
        torch.cuda.nvtx.range_pop()
    memory = {"before": before,
              "after_allocated_bytes": torch.cuda.memory_allocated(worker.device),
              "after_reserved_bytes": torch.cuda.memory_reserved(worker.device),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(worker.device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(worker.device)}
    memory["incremental_peak_allocated_bytes"] = memory["peak_allocated_bytes"]-before["allocated_bytes"]
    memory["incremental_peak_reserved_bytes"] = memory["peak_reserved_bytes"]-before["reserved_bytes"]
    event_ms = start.elapsed_time(end)
    compilation = monitor.events[event_index:]
    monitor.context = {"stage": "validation", "phase": case["phase"], "length": case["length"]}
    get_tp_group().barrier()  # Both ranks finish before validation or capture stop.
    checkpoint.verify_metadata(state.scalar_metadata())
    checkpoint.verify_write_boundaries()
    correct = {"passed": True, "logits_present": output is not None,
               "state_metadata_fixed": True, "write_boundaries_verified": True,
               "comparison": "byte_exact", "finite": None, "sha256": None}
    if case["output_presence"] is None:
        case["output_presence"] = output is not None
    correct["output_rank_stable"] = case["output_presence"] == (output is not None)
    correct["passed"] = correct["output_rank_stable"]
    if output is not None:
        cpu = output.detach().cpu().contiguous()
        correct["finite"] = bool(torch.isfinite(cpu).all())
        correct["sha256"] = tensor_fingerprint(cpu)
        reference = case["reference"]
        if reference is None:
            case["reference"] = cpu.clone()
            correct["reference_established"] = True
        else:
            correct["reference_established"] = False
            correct["byte_exact"] = tensor_fingerprint(reference) == correct["sha256"]
            correct["max_abs_difference"] = float((cpu.float()-reference.float()).abs().max()) if correct["finite"] else None
            correct["passed"] = correct["passed"] and correct["byte_exact"]
        correct["passed"] = correct["passed"] and correct["finite"]
    return {"rank": worker.rank, "stage": stage, "iteration": iteration, "attempt": attempt,
            "wall_seconds": elapsed, "cuda_event_ms": event_ms, "memory": memory,
            "correctness": correct, "compilation_events": compilation,
            "capture_active": case["capture_active"], "nvtx_range": label}


def worker_evidence(worker):
    case = worker._context_case
    return {"rank": worker.rank, "state": case["state"].evidence(),
            "checkpoint": case["checkpoint"].evidence(),
            "compilation_monitor": worker._context_monitor.coverage}


def summarize_attempt(attempt, instrumented):
    measurements = attempt["measurements"]
    if len(attempt["warmups"]) != WARMUPS or len(measurements) != MEASUREMENTS:
        return {"accepted": False, "reason": "incomplete measurement set"}
    correct = all(row["correctness_passed"] for row in attempt["warmups"] + measurements)
    capture_matches = all(r["capture_active"] is instrumented for m in measurements for r in m["ranks"])
    warmup_clean = not any(row["compilation_observed"] for row in attempt["warmups"])
    compiled = any(row["compilation_observed"] for row in measurements)
    result = {"accepted": correct and capture_matches and warmup_clean and not compiled, "correctness_passed": correct,
              "capture_state_matches": capture_matches, "warmup_compilation_free": warmup_clean,
              "compilation_observed": compiled,
              "two_rank_completion_seconds": stats([r["two_rank_completion_seconds"] for r in measurements]),
              "controller_rpc_seconds": stats([r["controller_rpc_seconds"] for r in measurements]),
              "per_rank": []}
    for rank in (0, 1):
        rows = [validate_ranks(m["ranks"])[rank] for m in measurements]
        result["per_rank"].append({"rank": rank,
            "wall_seconds": stats([r["wall_seconds"] for r in rows]),
            "cuda_event_ms": stats([r["cuda_event_ms"] for r in rows]),
            "peak_allocated_bytes": max(r["memory"]["peak_allocated_bytes"] for r in rows),
            "peak_reserved_bytes": max(r["memory"]["peak_reserved_bytes"] for r in rows),
            "incremental_peak_allocated_bytes": max(r["memory"]["incremental_peak_allocated_bytes"] for r in rows),
            "incremental_peak_reserved_bytes": max(r["memory"]["incremental_peak_reserved_bytes"] for r in rows)})
    result["eligible_for_unprofiled_timing"] = result["accepted"] and not instrumented
    return result


def validate_final_state(rows, expected_invocations):
    rows = validate_ranks(rows)
    for row in rows:
        counts = row["checkpoint"]
        for key in ("restore_count", "verified_restore_count", "verified_write_boundary_count"):
            if counts.get(key) != expected_invocations:
                raise ValueError(f"Rank {row['rank']} missing exact state checks: {key}")
        monitor = row["compilation_monitor"]
        if any(monitor.get(key) is not True for key in
               ("native_jit_monitor_active", "native_jit_callback", "triton_autotuner_bench")):
            raise ValueError("Required compilation observer not active")
    return rows


def run_case(call, phase, length, ids, instrumented, persist, max_attempts=2, rank_artifact_dir=None):
    """Controller orchestration is CPU-testable; call always means native RPC."""
    case = {"kind": "inkling-context-case-v1", "case_id": f"{phase}-L{length}",
            "phase": phase, "length": length, "query_tokens": length if phase == "prefill" else 1,
            "history_tokens": 0 if phase == "prefill" else length,
            "total_sequence_tokens": length if phase == "prefill" else length+1,
            "input_prefix_sha256": hash_ids(ids[:length if phase == "prefill" else length+1]),
            "instrumented": instrumented, "execution_mode": "eager", "status": "preparing",
            "numerical_policy": NUMERICAL_POLICY, "attempts": [], "preparation": []}
    persist(case)
    try:
        case["workers"] = validate_ranks(call(worker_prepare, phase, length, ids, rank_artifact_dir))
        # A separate explicit preparation forward establishes the output
        # reference and compiles this exact shape before five counted warmups.
        def invoke(stage, iteration, attempt):
            begin = time.perf_counter()
            rows = call(worker_iteration, stage, iteration, attempt)
            return combine_iteration(rows, time.perf_counter()-begin)
        case["preparation"].append(invoke("preparation", 0, 0))
        for number in range(max_attempts):
            attempt = {"index": number, "warmups": [], "warmup_preparation_rounds": [], "measurements": [], "capture": []}
            case["attempts"].append(attempt)
            case["status"] = "warming"
            for warmup_round in range(3):
                for iteration in range(WARMUPS):
                    attempt["warmups"].append(invoke("warmup", warmup_round*WARMUPS+iteration, number)); persist(case)
                if not any(row["compilation_observed"] for row in attempt["warmups"]):
                    break
                if warmup_round < 2:
                    attempt["warmup_preparation_rounds"].append(attempt["warmups"])
                    attempt["warmups"] = []
            if any(row["compilation_observed"] for row in attempt["warmups"]):
                raise RuntimeError("Compilation/autotuning persisted through three complete warmup rounds")
            capture_started = False
            try:
                if instrumented:
                    attempt["capture"].append(validate_ranks(call(worker_capture, True)))
                    capture_started = True
                case["status"] = "measuring"
                for iteration in range(MEASUREMENTS):
                    attempt["measurements"].append(invoke("measured", iteration, number)); persist(case)
            finally:
                if capture_started:
                    attempt["capture"].append(validate_ranks(call(worker_capture, False)))
            attempt["summary"] = summarize_attempt(attempt, instrumented)
            persist(case)
            if attempt["summary"]["accepted"] or not attempt["summary"]["compilation_observed"]:
                break
            # Preserve every sample. Five new warmups precede the complete
            # repeated ten-sample set; never remove individual slow calls.
        expected = len(case["preparation"]) + sum(len(a["warmups"])+len(a["measurements"])
                   + sum(len(r) for r in a["warmup_preparation_rounds"]) for a in case["attempts"])
        case["final_state"] = validate_final_state(call(worker_evidence), expected)
        accepted = case["attempts"][-1]["summary"]["accepted"] and all(r["correctness_passed"] for r in case["preparation"])
        case["status"] = "completed" if accepted else "incomplete"
        case["accepted_attempt"] = case["attempts"][-1]["index"] if accepted else None
        case["eligible_for_unprofiled_timing"] = accepted and not instrumented
    except Exception as exc:
        case["status"] = "failed"
        case["eligible_for_unprofiled_timing"] = False
        case["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        persist(case)
        raise
    persist(case)
    return case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/context-plan"))
    parser.add_argument("--lengths", default="256,1024,4096")
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--profile", action="store_true", help="ONLY instrumented cases, in a separate Nsight child")
    args = parser.parse_args(argv)
    if tuple(map(int, args.lengths.split(","))) != LENGTHS:
        parser.error("This fixed experiment requires --lengths 256,1024,4096 without shape substitution")
    config = {"engine_arguments": runtime.engine_arguments(), "boundary_id": runtime.BOUNDARY_ID,
              "model": runtime.MODEL, "revision": runtime.REVISION, "base_image": runtime.BASE_IMAGE,
              "baseline_run": runtime.BASELINE_RUN, "lengths": LENGTHS, "seed": args.seed,
              "warmups": WARMUPS, "measurements": MEASUREMENTS, "batch_size": 1, "concurrency": 1,
              "instrumented": args.profile, "execution_mode": "eager", "numerical_policy": NUMERICAL_POLICY,
              "timing_boundary": "Native full forward including layers/cache writes/final norm and last-token LM head; excludes metadata/reset/output validation",
              "two_rank_metric": "Maximum of rank-local synchronized wall times following a native CPU TP barrier; residual start skew remains; never a sum",
              "event_metric": "Rank current-stream CUDA event interval, including native host dispatch gaps and communication; not kernel-only duration",
              "capture_scope": "Per-case capture begins after five warmups; only context/*/measured/* NVTX ranges are analyzed; reset/validation copies occur between ranges",
              "compilation_policy": "Keep all10 on measured JIT/autotune; five new warmups then repeat all10 once. No selective rejection.",
              "limitations": ["One pretrained inference model; no backward/optimizer/Table1-size compliance", "4096 is not a maximum fitting context", "Native sequence parallelism changes at2048 input tokens", "No measured HBM bandwidth or saturation counters", "Cross-path full-prefill versus decode numerical equivalence remains additional pending validation"]}
    if not args.run:
        print(json.dumps({"status": "plan_only", **config}, indent=2)); return 0
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty; existing evidence is immutable")
    out.mkdir(parents=True, exist_ok=True)
    ids, inputs = input_sequence(args.seed)
    write_json(out/"config.json", config); write_json(out/"inputs.json", inputs)
    summary = {"kind": "inkling-context-sweep-v1", "status": "starting", "completed": False,
               "instrumented": args.profile, "case_count": 6, "completed_case_count": 0, "cases": []}
    write_json(out/"summary.json", summary)
    engine = None
    try:
        begin = time.perf_counter(); engine = runtime.create_engine(); startup = time.perf_counter()-begin
        workers = validate_ranks(runtime.collective(engine, worker_initialize))
        if any("B300" not in w["device_name"] for w in workers) or len({w["device_uuid"] for w in workers}) != 2:
            raise RuntimeError("Native workers are not on two distinct B300 GPUs")
        write_json(out/"runtime.json", {"kind": "inkling-context-runtime-v1", "model_loaded": True,
                   "workers": workers, "engine_arguments": runtime.engine_arguments(), "model_startup_seconds": startup})
        call = lambda fn, *a: runtime.collective(engine, fn, *a)
        for phase in ("prefill", "decode"):
            for length in LENGTHS:
                path = out/f"case-{phase}-L{length}.json"
                entry = {"case_id": f"{phase}-L{length}", "phase": phase, "length": length,
                         "status": "preparing", "eligible_for_unprofiled_timing": False, "result_file": path.name}
                summary["cases"].append(entry)
                def persist(case):
                    write_json(path, case)
                    entry.update(status=case["status"], eligible_for_unprofiled_timing=case.get("eligible_for_unprofiled_timing", False))
                    write_json(out/"summary.json", summary)
                result = run_case(call, phase, length, ids, args.profile, persist,
                                  rank_artifact_dir=str(out/"worker-observations"/entry["case_id"]))
                summary["completed_case_count"] += result["status"] == "completed"
                write_json(out/"summary.json", summary)
        summary["completed"] = summary["completed_case_count"] == 6
        summary["status"] = "completed" if summary["completed"] else "incomplete"
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        if engine is not None:
            begin = time.perf_counter()
            try:
                engine.shutdown()
                summary["engine_shutdown_completed"] = True
            except Exception as exc:
                summary["engine_shutdown_completed"] = False
                summary["shutdown_error"] = str(exc)
                summary["completed"] = False; summary["status"] = "failed"
            summary["shutdown_seconds"] = time.perf_counter()-begin
        write_json(out/"summary.json", summary)
        manifest(out, kind="inkling-context-sweep", status=summary["status"], metadata={"instrumented": args.profile})
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
