"""Resident native/fused relative-bias A/B experiment; default is plan-only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

from profiler.artifacts import manifest, write_json
from experiments.context_sweep import benchmark as base, runtime
from . import adapter

BLOCKS = 2
WARMUP_PAIRS = 5
MEASURED_PAIRS = 10


def plan():
    return {"kind": "relative-bias-experiment-plan-v1", "schema_version": 1,
            "model": runtime.MODEL, "revision": runtime.REVISION, "base_image": runtime.BASE_IMAGE,
            "engine_arguments": runtime.engine_arguments(), "gpu": "B300:2", "seed": 336,
            "variants": ["native", "fused"], "execution_mode": "eager",
            "cases": [{"phase": phase, "length": length} for phase in ("prefill", "decode") for length in base.LENGTHS],
            "confirmation_blocks": BLOCKS, "warmup_pairs_per_block": WARMUP_PAIRS,
            "measured_pairs_per_block": MEASURED_PAIRS,
            "pair_order": "Alternating native/fused order; starting order reversed in block1. Same checkpoint restored before each variant.",
            "compilation_policy": "Retain whole10-pair contaminated attempt; repeat5clean warmup pairs and all10 pairs once within the same block. Retain warmup rounds with compilation.",
            "correctness": "Adversarial and CUDAgraph metadata replay helper tests plus captured actual inputs must pass before measured pairs. Full-model finite logits compare byte-exactly to one native reference retained across both variants.",
            "timing": "Native forward includes full layers/cache writes/final norm/last-token LM head. Synchronized rank-local wall and CUDA events; model latency=max(rank walls), never sum. Reset/barriers/validation excluded.",
            "helper_microbenchmark": {"execution_mode": "eager", "warmups": 5, "measured_iterations": 10, "repetitions_per_sample": 20,
                "scope": "Separate from full-model latency; API/event intervals include host launch gaps, not kernel-only cost."},
            "source_sha256": adapter.ATTENTION_SHA256, "original_helper_sha256": adapter.HELPER_SHA256,
            "patch_scope": "Only _unshear_rel_logits global dispatch; _use_sheared_fa4 remains the original False B300 fallback. Native/fused share the same wrapper dispatch/counting overhead.",
            "capture_bytes_per_rank": adapter.CAPTURE_BYTES_PER_RANK,
            "instrumentation": "One plain resident A/B child and one separate Nsight resident A/B child. No model reload between cases/variants within each child; no Nsight-attached timing called unprofiled.",
            "memory_caveat": "Live allocations and fixed reserved/cache/workspace pools are separate. Resident allocator/JIT state persists across variants; peaks are not a proof of total deployment capacity savings.",
            "predeclared_adoption_gate": {"primary": "prefill-L4096", "primary_required": "Both confirmation-block native/fused mean-latency ratios>1; pooled paired-bootstrap95% CI lower bound>1",
                "decode_required": "All three decode fused means<=native means; no observed decode regression",
                "correctness_required": "All exact helper, graph, captured-input, full-model and state gates pass",
                "serving_validation": "Separate required gate after eager winner; this experiment alone does not establish a serving speedup"}}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pair_order(iteration, block):
    return ("native", "fused") if (iteration+block) % 2 == 0 else ("fused", "native")


def combine_pair(observations, order, instrumented, *, require_helper=False):
    if set(observations) != {"native", "fused"}:
        raise ValueError("A complete native/fused pair is required")
    passed = all(item["correctness_passed"] for item in observations.values())
    counts_match = True
    reference_match = True
    for rank in (0, 1):
        rows = {name: base.validate_ranks(value["ranks"])[rank] for name, value in observations.items()}
        counts_match &= rows["native"].get("helper_calls") == rows["fused"].get("helper_calls")
        if require_helper:
            counts_match &= rows["native"].get("helper_calls", 0) > 0
        if rows["native"]["correctness"]["logits_present"]:
            refs = [r["correctness"].get("reference_sha256") for r in rows.values()]
            reference_match &= bool(refs[0]) and refs[0] == refs[1]
    capture_matches = all(row["capture_active"] is instrumented for item in observations.values() for row in item["ranks"])
    return {"order": list(order), "observations": observations,
            "correctness_passed": passed and counts_match and reference_match,
            "helper_counts_match": counts_match, "native_reference_matches": reference_match,
            "capture_state_matches": capture_matches,
            "compilation_observed": any(v["compilation_observed"] for v in observations.values()),
            "native_over_fused_wall_ratio": observations["native"]["two_rank_completion_seconds"]/observations["fused"]["two_rank_completion_seconds"]}


def summarize_attempt(attempt, instrumented):
    warmups, measured = attempt["warmups"], attempt["measurements"]
    complete = len(warmups) == WARMUP_PAIRS and len(measured) == MEASURED_PAIRS
    correct = complete and all(p["correctness_passed"] for p in warmups+measured)
    compiled = any(p["compilation_observed"] for p in measured)
    clean_warmups = not any(p["compilation_observed"] for p in warmups)
    capture_ok = all(p["capture_state_matches"] for p in measured)
    result = {"accepted": correct and not compiled and clean_warmups and capture_ok,
              "correctness_passed": correct, "compilation_observed": compiled,
              "warmups_compilation_free": clean_warmups, "capture_state_matches": capture_ok}
    if measured:
        result["variants"] = {name: {"two_rank_completion_seconds": base.stats([p["observations"][name]["two_rank_completion_seconds"] for p in measured]),
            "per_rank": [{"rank": rank,
                          "wall_seconds": base.stats([base.validate_ranks(p["observations"][name]["ranks"])[rank]["wall_seconds"] for p in measured]),
                          "cuda_event_ms": base.stats([base.validate_ranks(p["observations"][name]["ranks"])[rank]["cuda_event_ms"] for p in measured])}
                         for rank in (0, 1)]} for name in ("native", "fused")}
    result["eligible_for_unprofiled_timing"] = result["accepted"] and not instrumented
    return result


def run_case(call, phase, length, ids, out, instrumented, persist):
    case = {"kind": "relative-bias-case-v1", "case_id": f"{phase}-L{length}", "phase": phase,
            "length": length, "history_tokens": length if phase == "decode" else 0,
            "query_tokens": 1 if phase == "decode" else length, "instrumented": instrumented,
            "status": "preparing", "preparation": [], "blocks": [], "eligible_for_unprofiled_timing": False}
    invocation_count = 0
    def invoke(variant, stage, iteration, attempt, block, capture=False):
        nonlocal invocation_count
        start = time.perf_counter()
        rows = call(adapter.worker_iteration, variant, stage, iteration, attempt, block, capture)
        invocation_count += 1
        return base.combine_iteration(rows, time.perf_counter()-start)
    def pair(stage, iteration, attempt, block):
        order = pair_order(iteration, block)
        observations = {name: invoke(name, stage, iteration, attempt, block) for name in order}
        return combine_pair(observations, order, instrumented if stage == "measured" else False, require_helper=phase == "prefill")
    persist(case)
    try:
        case["prepared_workers"] = base.validate_ranks(call(adapter.worker_prepare, phase, length, ids, str(out)))
        case["preparation"].append(invoke("native", "preparation", 0, 0, 0, True)); persist(case)
        case["captured_input_correctness"] = base.validate_ranks(call(adapter.worker_helper_validation, str(out)))
        if not all(row.get("passed") is True or (phase == "decode" and row.get("applicable") is False)
                   for row in case["captured_input_correctness"]):
            raise RuntimeError("Actual captured helper correctness gate failed")
        case["preparation"].append(invoke("fused", "preparation", 1, 0, 0)); persist(case)
        if not all(row["correctness_passed"] for row in case["preparation"]):
            raise RuntimeError("Native/fused preparation logits or state did not match")
        case["helper_microbenchmark"] = base.validate_ranks(call(adapter.worker_helper_microbench, str(out)))
        for block_index in range(BLOCKS):
            block = {"block_index": block_index, "attempts": [], "accepted_attempt": None}
            case["blocks"].append(block)
            for attempt_index in range(2):
                attempt = {"attempt": attempt_index, "warmup_preparation_rounds": [], "warmups": [], "measurements": [], "capture": []}
                block["attempts"].append(attempt)
                case["status"] = "warming"
                for warmup_round in range(3):
                    for iteration in range(WARMUP_PAIRS):
                        attempt["warmups"].append(pair("warmup", warmup_round*WARMUP_PAIRS+iteration, attempt_index, block_index)); persist(case)
                    if not any(p["compilation_observed"] for p in attempt["warmups"]):
                        break
                    if warmup_round < 2:
                        attempt["warmup_preparation_rounds"].append(attempt["warmups"]); attempt["warmups"] = []
                if any(p["compilation_observed"] for p in attempt["warmups"]):
                    raise RuntimeError("Compilation persisted through three full warmup-pair rounds")
                started = False
                try:
                    if instrumented:
                        attempt["capture"].append(base.validate_ranks(call(base.worker_capture, True))); started = True
                    case["status"] = "measuring"
                    for iteration in range(MEASURED_PAIRS):
                        attempt["measurements"].append(pair("measured", iteration, attempt_index, block_index)); persist(case)
                finally:
                    if started:
                        attempt["capture"].append(base.validate_ranks(call(base.worker_capture, False)))
                attempt["summary"] = summarize_attempt(attempt, instrumented); persist(case)
                if attempt["summary"]["accepted"]:
                    block["accepted_attempt"] = attempt_index; break
                if not attempt["summary"]["compilation_observed"]:
                    break
            block["completed"] = block["accepted_attempt"] is not None
            persist(case)
        case["final_state"] = base.validate_final_state(call(adapter.worker_evidence), invocation_count)
        case["completed"] = len(case["blocks"]) == BLOCKS and all(b["completed"] for b in case["blocks"])
        case["status"] = "completed" if case["completed"] else "incomplete"
        case["eligible_for_unprofiled_timing"] = case["completed"] and not instrumented
    except Exception as exc:
        case.update(status="failed", completed=False, eligible_for_unprofiled_timing=False,
                    error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        persist(case); raise
    persist(case)
    return case


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args(argv)
    protocol = plan()
    if args.plan and canonical_hash(json.loads(args.plan.read_text())) != canonical_hash(protocol):
        parser.error("Provided frozen plan does not exactly match this harness protocol")
    if not args.run:
        print(json.dumps(protocol, indent=2)); return 0
    if not args.output_dir:
        parser.error("--output-dir is required; existing artifacts may not be overwritten")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    ids, inputs = base.input_sequence(protocol["seed"])
    write_json(out/"config.json", {**protocol, "instrumented": args.profile, "plan_sha256": canonical_hash(protocol)})
    write_json(out/"inputs.json", inputs)
    summary = {"kind": "relative-bias-sweep-v1", "status": "starting", "completed": False,
               "plan_sha256": canonical_hash(protocol),
               "instrumented": args.profile, "case_count": 6, "completed_case_count": 0, "cases": [],
               "correctness_passed": False, "engine_start_count": 0, "weights_retained_between_variants": True,
               "adoption_status": "not_determined_by_harness; predeclared timing and serving gates require separate analysis"}
    write_json(out/"summary.json", summary)
    engine = None
    try:
        from experiments.context_sweep.inventory import runtime_inventory
        write_json(out/"runtime-inventory.json", runtime_inventory(phase="relative-bias-profile-child" if args.profile else "relative-bias-plain-child", gpu_requested=True))
        begin = time.perf_counter(); engine = runtime.create_engine(); startup = time.perf_counter()-begin
        summary["engine_start_count"] = 1
        call = lambda fn, *a: runtime.collective(engine, fn, *a)
        workers = base.validate_ranks(call(adapter.worker_initialize, str(out), args.profile))
        if not all(r.get("adversarial_correctness", {}).get("passed") is True for r in workers):
            raise RuntimeError("Adversarial/graph helper correctness gate failed")
        if any("B300" not in r["device_name"] for r in workers) or len({r["device_uuid"] for r in workers}) != 2:
            raise RuntimeError("Two distinct B300 worker identities required")
        summary["model_loaded"] = True
        identity = {r["rank"]: (r["worker_pid"], r["device_uuid"]) for r in workers}
        write_json(out/"runtime.json", {"kind": "relative-bias-runtime-v1", "model_loaded": True, "workers": workers,
                   "engine_arguments": runtime.engine_arguments(), "engine_start_count": 1,
                   "weights_retained_between_variants": True, "model_startup_seconds": startup})
        for item in protocol["cases"]:
            phase, length = item["phase"], item["length"]
            entry = {"case_id": f"{phase}-L{length}", "phase": phase, "length": length,
                     "status": "preparing", "result_file": f"case-{phase}-L{length}.json"}
            summary["cases"].append(entry)
            def persist(case):
                write_json(out/entry["result_file"], case)
                entry.update(status=case["status"], eligible_for_unprofiled_timing=case["eligible_for_unprofiled_timing"])
                write_json(out/"summary.json", summary)
            case = run_case(call, phase, length, ids, out, args.profile, persist)
            for row in case["final_state"]:
                if (row["worker_pid"], row["device_uuid"]) != identity[row["rank"]]:
                    raise RuntimeError("Worker/device identity changed between variants or cases")
            summary["completed_case_count"] += case["status"] == "completed"
        summary["completed"] = summary["completed_case_count"] == 6
        summary["correctness_passed"] = summary["completed"]
        summary["status"] = "completed" if summary["completed"] else "incomplete"
    except Exception as exc:
        summary.update(status="failed", completed=False, error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    finally:
        if engine is not None:
            try:
                summary["patch_restoration"] = base.validate_ranks(runtime.collective(engine, adapter.worker_restore))
                if not all(r["helper_patch"]["restored"] for r in summary["patch_restoration"]):
                    raise RuntimeError("Original helper restoration was not proven")
            except Exception as exc:
                summary.update(status="failed", completed=False, restoration_error=str(exc))
            begin = time.perf_counter()
            try:
                engine.shutdown(); summary["engine_shutdown_completed"] = True
            except Exception as exc:
                summary.update(status="failed", completed=False, engine_shutdown_completed=False, shutdown_error=str(exc))
            summary["shutdown_seconds"] = time.perf_counter()-begin
        summary["eligible_for_unprofiled_timing"] = summary["completed"] and not args.profile
        write_json(out/"summary.json", summary)
        manifest(out, kind="relative-bias-experiment", status=summary["status"], metadata={"instrumented": args.profile})
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
