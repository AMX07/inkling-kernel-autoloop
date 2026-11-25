"""Opt-in future experiment batches retaining one native TP2 engine.

uv run --no-project python -m experiments.context_sweep.resident_session
uv run --no-project python -m experiments.context_sweep.resident_session --plan PLAN.json --run --output-dir NEW_DIRECTORY

Default behavior prints a plan without CUDA/Modal work. Execute only inside the
pinned runtime under an external allocation deadline. All experiments share one
instrumentation policy: a session with Nsight attached is never unprofiled.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import traceback
import uuid

from profiler.artifacts import manifest, write_json
from . import benchmark, runtime

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = ("experiments/context_sweep/resident_session.py", "experiments/context_sweep/benchmark.py",
                "experiments/context_sweep/runtime.py", "experiments/context_sweep/state.py",
                "experiments/context_sweep/inventory.py", "profiler/artifacts.py", "AGENTS.md",
                "planning/resident-session-policy.json", "pyproject.toml", "uv.lock", ".python-version")
SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
ATTACHMENT_MARKERS = ("libtoolsinjection", "libnsys")
TOOLING_LIBRARIES = ("libcupti", "libnvperf")
ATTACHMENT_ENV = ("NSYS_PROFILING_SESSION_ID", "NSYS_SYSTEM_PROFILING_SESSION_ID", "NSYS_INJECTION_SERVER_PORT")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_plan():
    return {"schema_version": 1, "kind": "native-resident-session-plan", "session_label": "native-context-batch",
            "instrumentation": "none", "max_session_seconds": 1800,
            "engine_arguments": runtime.engine_arguments(),
            "experiments": [{"id": phase+"-seed336", "variant": "native", "seed": 336,
                             "cases": [{"phase": phase, "length": length} for length in benchmark.LENGTHS]}
                            for phase in ("prefill", "decode")]}


def validate_plan(plan):
    # Reject unsupported execution/config fields rather than silently ignoring a
    # requested variant, profiler transition or engine mutation.
    expected = {"schema_version", "kind", "session_label", "instrumentation", "max_session_seconds", "engine_arguments", "experiments"}
    if not isinstance(plan, dict) or set(plan) != expected or plan["schema_version"] != 1 or plan["kind"] != "native-resident-session-plan":
        raise ValueError("Unknown resident-session plan schema or fields")
    if not isinstance(plan["session_label"], str) or not SLUG.fullmatch(plan["session_label"]):
        raise ValueError("Session label must be a safe lowercase identifier")
    if plan["instrumentation"] not in ("none", "nsys"):
        raise ValueError("Instrumentation must be session-wide none or nsys; mixed sessions are unsupported")
    if type(plan["max_session_seconds"]) is not int or not 1 <= plan["max_session_seconds"] <= 1800:
        raise ValueError("Session deadline must be an integer in1..1800 seconds")
    if canonical_hash(plan["engine_arguments"]) != canonical_hash(runtime.engine_arguments()):
        raise ValueError("Resident batch requires the exact pinned native engine configuration")
    experiments = plan["experiments"]
    if not isinstance(experiments, list) or not 1 <= len(experiments) <= 16:
        raise ValueError("Declare1..16 bounded experiments")
    seen = set(); count = 0
    for spec in experiments:
        if not isinstance(spec, dict) or set(spec) != {"id", "variant", "seed", "cases"}:
            raise ValueError("Unknown experiment fields; per-experiment engine/profiler changes are not supported")
        if not isinstance(spec["id"], str) or not SLUG.fullmatch(spec["id"]) or spec["id"] in seen:
            raise ValueError("Experiment IDs must be unique safe lowercase identifiers")
        seen.add(spec["id"])
        if spec["variant"] != "native":
            raise ValueError("Only the native variant is implemented; kernel replacements need an independently reviewed adapter")
        if type(spec["seed"]) is not int or not 0 <= spec["seed"] < 2**32:
            raise ValueError("Seed must be an unsigned32-bit integer")
        if not isinstance(spec["cases"], list) or not 1 <= len(spec["cases"]) <= 6:
            raise ValueError("Each experiment must declare1..6 cases")
        cases = set()
        for case in spec["cases"]:
            if (not isinstance(case, dict) or set(case) != {"phase", "length"}
                    or case["phase"] not in ("prefill", "decode")
                    or type(case["length"]) is not int or case["length"] not in benchmark.LENGTHS):
                raise ValueError("Unsupported phase/length; no silent substitution")
            key = (case["phase"], case["length"])
            if key in cases:
                raise ValueError("Duplicate cases require a separately identified experiment")
            cases.add(key)
        count += len(cases)
    if count > 48:
        raise ValueError("Resident batch exceeds48 declared cases")
    return json.loads(json.dumps(plan))


def profiler_attachment_probe(*, environment=None, maps_text=None):
    """Conservative Linux injection check, without exporting environment values.

    This cannot detect every external profiler. Plain-session eligibility also
    requires the launcher's declared no-profiler policy. CUPTI/NVPerf library
    presence alone is recorded separately: a dependency can be loaded without
    an active profiler, so that alone cannot prove attachment.
    """
    environment = os.environ if environment is None else environment
    available = maps_text is not None
    if maps_text is None:
        try:
            maps_text = Path("/proc/self/maps").read_text(); available = True
        except OSError:
            maps_text = ""
    text = maps_text.lower()
    markers = [name for name in ATTACHMENT_MARKERS if name in text]
    env_markers = [key for key in ATTACHMENT_ENV if environment.get(key)]
    preload = environment.get("LD_PRELOAD", "").lower()
    if any(marker in preload for marker in ATTACHMENT_MARKERS):
        env_markers.append("LD_PRELOAD_profiler_marker")
    return {"maps_checked": available, "known_profiler_detected": bool(markers or env_markers),
            "library_markers": markers, "environment_marker_names": env_markers,
            "tooling_libraries_loaded": [name for name in TOOLING_LIBRARIES if name in text],
            "coverage_limit": "Known Linux injection markers only; no universal proof that an external profiler is absent"}


def worker_session_identity(worker):
    value = runtime.worker_runtime_summary(worker)
    value["worker_pid"] = os.getpid()
    value["profiler_probe"] = profiler_attachment_probe()
    return value


def check_identity(workers, instrumentation, expected=None):
    rows = benchmark.validate_ranks(workers)
    if any("B300" not in r.get("device_name", "") for r in rows) or len({r.get("device_uuid") for r in rows}) != 2:
        raise ValueError("Resident session requires two distinct B300 devices")
    identity = [{key: r.get(key) for key in ("rank", "device_uuid", "worker_pid")} for r in rows]
    if any(not r["device_uuid"] or type(r["worker_pid"]) is not int or r["worker_pid"] <= 0 for r in identity):
        raise ValueError("Actual worker identity is incomplete")
    if expected is not None and identity != expected:
        raise ValueError("Resident worker or device changed between experiments")
    if instrumentation == "none":
        if any(r.get("profiler_probe", {}).get("maps_checked") is not True
               or r["profiler_probe"].get("known_profiler_detected") is not False for r in rows):
            raise ValueError("Plain timing refused: profiler attachment detected or Linux probe unavailable")
    return identity


def freeze_sources(out, *, root=ROOT):
    hashes = {}
    for relative in SOURCE_FILES:
        source = root/relative
        if source.is_symlink() or not source.is_file():
            raise ValueError("Missing or unsafe resident-session source")
        content = source.read_bytes()
        hashes[relative] = hashlib.sha256(content).hexdigest()
        target = out/"executed-source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    write_json(out/"source-provenance.json", {"kind": "resident-session-source-provenance-v1", "files": hashes,
        "baseline_run": runtime.BASELINE_RUN, "model": runtime.MODEL, "revision": runtime.REVISION,
        "base_image": runtime.BASE_IMAGE, "note": "Session code snapshot; launcher remains responsible for frozen image compatibility-layer provenance."})
    return hashes


def assert_sources_unchanged(hashes, *, root=ROOT):
    for relative, digest in hashes.items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("Resident session source changed after freezing")


def run_session(plan, out, *, create_engine=None, collective=None, case_runner=None, capture_inventory=None, source_root=ROOT):
    """CPU-testable coordination; actual default execution uses native callbacks."""
    plan = validate_plan(plan)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use an empty result directory; resident-session evidence is immutable")
    out.mkdir(parents=True, exist_ok=True)
    frozen = freeze_sources(out, root=source_root)
    write_json(out/"plan.json", plan)
    create_engine = create_engine or runtime.create_engine
    collective = collective or runtime.collective
    case_runner = case_runner or benchmark.run_case
    if capture_inventory is None:
        from .inventory import runtime_inventory
        capture_inventory = lambda: runtime_inventory(phase="resident-session-start", gpu_requested=True)
    start = time.monotonic()
    deadline = start+plan["max_session_seconds"]
    identity = None
    engine = None
    completed_cases = 0
    ledger = {"kind": "native-resident-session-v1", "session_uuid": str(uuid.uuid4()),
              "session_label": plan["session_label"], "plan_sha256": canonical_hash(plan),
              "engine_config_sha256": canonical_hash(plan["engine_arguments"]),
              "started_at": datetime.now(timezone.utc).isoformat(), "status": "starting", "completed": False,
              "instrumentation": plan["instrumentation"], "engine_start_count": 0,
              "eligible_for_unprofiled_timing": False,
              "weights_retention_policy": "Retain across every compatible experiment until the whole declared session ends",
              "weights_retained_between_experiments": None, "experiments": [],
              "notes": ["Native cache/input state is reset for every case and iteration; compiled kernels, allocator pools and communication protocol counters persist.",
                        "Weights are unloaded only at the end of the whole declared session, never between compatible experiments.",
                        "The allocation owner must reuse a live compatible session before starting another allocation. This batch driver does not allocate GPUs or provide a persistent service endpoint.",
                        "Order/cache-temperature effects are not randomized away; experiment order is preserved in the ledger.",
                        "A session with Nsight attached has no unprofiled headline timings, including capture-inactive intervals.",
                        "Case timing flags are conditional on this session's completed, identity-checked and instrumentation-checked eligibility gate.",
                        "External launcher allocation timeout is required; internal deadlines are checked between native operations."]}
    def save():
        write_json(out/"session-ledger.json", ledger)
    def call(function, *args):
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Resident session deadline reached; no next experiment is started")
        return collective(engine, function, *args, timeout=min(remaining, 1200))
    save()
    try:
        inventory_path = out/"runtime-inventory.json"
        write_json(inventory_path, capture_inventory())
        inventory_hash = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        before = time.perf_counter()
        engine = create_engine()
        ledger["engine_start_count"] = 1
        ledger["model_startup_seconds"] = time.perf_counter()-before
        call(benchmark.worker_initialize)
        workers = call(worker_session_identity)
        identity = check_identity(workers, plan["instrumentation"])
        ledger["resident_identity"] = identity
        ledger["model_loaded"] = True
        write_json(out/"runtime.json", {"kind": "resident-session-runtime-v1", "workers": workers,
                   "engine_arguments": plan["engine_arguments"], "model_startup_seconds": ledger["model_startup_seconds"]})
        ledger["status"] = "running"; save()
        for order, spec in enumerate(plan["experiments"]):
            assert_sources_unchanged(frozen, root=source_root)
            workers = call(worker_session_identity)
            check_identity(workers, plan["instrumentation"], identity)
            directory = out/"experiments"/spec["id"]
            directory.mkdir(parents=True, exist_ok=False)
            entry = {"id": spec["id"], "order": order, "status": "running", "result_directory": str(directory.relative_to(out)),
                     "resident_identity_verified": True, "case_count": len(spec["cases"]), "completed_case_count": 0,
                     "engine_reloaded": False, "cases": []}
            ledger["experiments"].append(entry); save()
            if order > 0:
                ledger["weights_retained_between_experiments"] = True
            ids, input_policy = benchmark.input_sequence(spec["seed"])
            write_json(directory/"inputs.json", input_policy)
            write_json(directory/"frozen-config.json", {"kind": "resident-experiment-config-v1", "session_uuid": ledger["session_uuid"],
                "experiment": spec, "experiment_sha256": canonical_hash(spec), "plan_sha256": ledger["plan_sha256"],
                "engine_arguments": plan["engine_arguments"], "engine_config_sha256": ledger["engine_config_sha256"],
                "session_source_hashes": frozen, "session_executed_source": "../../executed-source",
                "session_runtime_inventory": "../../runtime-inventory.json", "session_runtime_inventory_sha256": inventory_hash,
                "resident_identity": identity, "input_sha256": input_policy["sha256"], "instrumentation": plan["instrumentation"],
                "warmups": benchmark.WARMUPS, "measurements": benchmark.MEASUREMENTS,
                "numerical_policy": benchmark.NUMERICAL_POLICY, "execution_mode": "eager"})
            for case in spec["cases"]:
                name = f"{case['phase']}-L{case['length']}"
                result_entry = {"case_id": name, "status": "preparing", "result_file": f"case-{name}.json"}
                entry["cases"].append(result_entry)
                def persist(result):
                    write_json(directory/result_entry["result_file"], result)
                    result_entry["status"] = result["status"]
                    result_entry["eligible_for_unprofiled_timing"] = result.get("eligible_for_unprofiled_timing", False)
                    save()
                result = case_runner(call, case["phase"], case["length"], ids,
                                     plan["instrumentation"] == "nsys", persist,
                                     rank_artifact_dir=str(directory/"worker-observations"/name))
                if result["status"] == "completed":
                    entry["completed_case_count"] += 1; completed_cases += 1
            # Verify retention and attachment policy after the case sequence as
            # well: an unexpected worker restart or injected profiler invalidates
            # the session's plain timing eligibility.
            check_identity(call(worker_session_identity), plan["instrumentation"], identity)
            assert_sources_unchanged(frozen, root=source_root)
            entry["status"] = "completed" if entry["completed_case_count"] == entry["case_count"] else "incomplete"
            entry["eligible_for_unprofiled_timing"] = entry["status"] == "completed" and plan["instrumentation"] == "none"
            write_json(directory/"summary.json", {"kind": "resident-experiment-summary-v1", "session_uuid": ledger["session_uuid"], **entry})
            manifest(directory, kind="resident-experiment", status=entry["status"], metadata={"session_uuid": ledger["session_uuid"]})
            save()
        ledger["completed"] = (len(ledger["experiments"]) == len(plan["experiments"])
                               and all(e["status"] == "completed" for e in ledger["experiments"]))
        ledger["status"] = "completed" if ledger["completed"] else "incomplete"
    except Exception as exc:
        ledger["status"] = "failed"; ledger["completed"] = False
        ledger["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        if ledger["experiments"] and ledger["experiments"][-1]["status"] == "running":
            entry = ledger["experiments"][-1]; entry["status"] = "failed"
            directory = out/entry["result_directory"]
            write_json(directory/"summary.json", {"kind": "resident-experiment-summary-v1", "session_uuid": ledger["session_uuid"], **entry})
            manifest(directory, kind="resident-experiment", status="failed")
    finally:
        if engine is not None:
            before = time.perf_counter()
            try:
                engine.shutdown(); ledger["engine_shutdown_completed"] = True
            except Exception as exc:
                ledger["engine_shutdown_completed"] = False; ledger["shutdown_error"] = str(exc)
                ledger["status"] = "failed"; ledger["completed"] = False
            ledger["final_shutdown_seconds"] = time.perf_counter()-before
        count = sum(e["status"] == "completed" for e in ledger["experiments"])
        startup = ledger.get("model_startup_seconds")
        ledger["startup_accounting"] = {"completed_experiments": count, "completed_cases": completed_cases,
            "startup_seconds_per_completed_experiment": startup/count if startup is not None and count else None,
            "startup_seconds_per_completed_case": startup/completed_cases if startup is not None and completed_cases else None,
            "interpretation": "Accounting division of one observed model startup; not forward latency or an observed speedup."}
        ledger["elapsed_seconds"] = time.monotonic()-start
        ledger["finished_at"] = datetime.now(timezone.utc).isoformat()
        ledger["eligible_for_unprofiled_timing"] = ledger["completed"] and plan["instrumentation"] == "none"
        save(); manifest(out, kind="native-resident-session", status=ledger["status"])
    return ledger


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    plan = validate_plan(json.loads(args.plan.read_text()) if args.plan else default_plan())
    if not args.run:
        print(json.dumps({"status": "plan_only", "plan": plan}, indent=2)); return 0
    if not args.output_dir:
        parser.error("Explicit --output-dir is required for a new immutable resident session")
    ledger = run_session(plan, args.output_dir)
    print(json.dumps({"status": ledger["status"], "session_uuid": ledger["session_uuid"], "result_directory": str(args.output_dir)}))
    return 0 if ledger["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
