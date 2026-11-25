"""CPU-only resident-session control tests, not native GPU execution proof."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.context_sweep import resident_session as s


class Backend:
    def __init__(self, *, fail_case=None, restart_on_identity=None, profiler=False):
        self.starts = 0; self.stops = 0; self.initializations = 0; self.identities = 0
        self.calls = []; self.case_calls = []; self.fail_case = fail_case
        self.restart_on_identity = restart_on_identity; self.profiler = profiler
        self.engine = object()

    def create(self):
        self.starts += 1
        backend = self
        class Engine:
            def shutdown(self):
                backend.stops += 1
        self.engine = Engine()
        return self.engine

    def collective(self, engine, callback, *args, timeout):
        if engine is not self.engine:
            raise AssertionError("Engine was replaced")
        self.calls.append((callback.__name__, timeout))
        if callback is s.benchmark.worker_initialize:
            self.initializations += 1; return [{"rank": 0}, {"rank": 1}]
        if callback is not s.worker_session_identity:
            raise AssertionError("Unexpected callback in orchestration fixture")
        self.identities += 1
        restart = self.identities == self.restart_on_identity
        return [{"rank": rank, "device_name": "NVIDIA B300", "device_uuid": f"fixture-gpu-{rank}",
                 "worker_pid": 100+rank+(100 if restart else 0),
                 "profiler_probe": {"maps_checked": True, "known_profiler_detected": self.profiler}}
                for rank in (0, 1)]

    def case_runner(self, call, phase, length, ids, instrumented, persist, **kwargs):
        key = (phase, length)
        self.case_calls.append({"case": key, "input_hash": s.benchmark.hash_ids(ids), "instrumented": instrumented})
        result = {"kind": "cpu-orchestration-fixture-only", "status": "measuring"}
        persist(result)
        if key == self.fail_case:
            result["status"] = "failed"; persist(result)
            raise RuntimeError("fixture native callback failed")
        result.update(status="completed", eligible_for_unprofiled_timing=not instrumented)
        persist(result)
        return result


class ResidentSessionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for name in s.SOURCE_FILES:
            file = self.root/name; file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# CPU fixture source; not a GPU runtime\n")
        self.out = self.root/"session"

    def run_fixture(self, backend=None, plan=None):
        backend = backend or Backend()
        ledger = s.run_session(plan or s.default_plan(), self.out, create_engine=backend.create,
                               collective=backend.collective, case_runner=backend.case_runner,
                               capture_inventory=lambda: {"kind": "cpu-test-inventory-fixture"}, source_root=self.root)
        return ledger, backend

    def test_one_startup_and_one_final_shutdown_across_ordered_experiments(self):
        ledger, backend = self.run_fixture()
        self.assertTrue(ledger["completed"])
        self.assertEqual((backend.starts, backend.stops, backend.initializations), (1, 1, 1))
        self.assertEqual(len(backend.case_calls), 6)
        self.assertEqual([r["case"][0] for r in backend.case_calls], ["prefill"]*3+["decode"]*3)
        self.assertEqual(len({r["input_hash"] for r in backend.case_calls}), 1)
        self.assertTrue(ledger["eligible_for_unprofiled_timing"])
        self.assertTrue(ledger["weights_retained_between_experiments"])
        self.assertEqual(ledger["startup_accounting"]["completed_experiments"], 2)
        self.assertEqual(ledger["startup_accounting"]["completed_cases"], 6)
        self.assertEqual(ledger["startup_accounting"]["startup_seconds_per_completed_experiment"], ledger["model_startup_seconds"]/2)
        configs = list((self.out/"experiments").glob("*/frozen-config.json"))
        self.assertEqual(len(configs), 2)
        for path in configs:
            config = json.loads(path.read_text())
            self.assertEqual(config["session_uuid"], ledger["session_uuid"])
            self.assertEqual(config["resident_identity"], ledger["resident_identity"])
            self.assertEqual(config["session_source_hashes"], json.loads((self.out/"source-provenance.json").read_text())["files"])

    def test_later_failure_preserves_previous_experiment_and_shuts_down_once(self):
        ledger, backend = self.run_fixture(Backend(fail_case=("decode", 256)))
        self.assertFalse(ledger["completed"])
        self.assertEqual([e["status"] for e in ledger["experiments"]], ["completed", "failed"])
        self.assertEqual((backend.starts, backend.stops), (1, 1))
        self.assertEqual(len(backend.case_calls), 4)
        self.assertTrue((self.out/"experiments/prefill-seed336/manifest.json").exists())
        partial = json.loads((self.out/"experiments/decode-seed336/case-decode-L256.json").read_text())
        self.assertEqual(partial["status"], "failed")
        self.assertFalse(ledger["eligible_for_unprofiled_timing"])

    def test_worker_restart_rejects_resident_identity_and_does_not_start_next_cases(self):
        ledger, backend = self.run_fixture(Backend(restart_on_identity=4))
        self.assertFalse(ledger["completed"])
        self.assertIn("changed between experiments", ledger["error"]["message"])
        self.assertEqual(len(backend.case_calls), 3)
        self.assertEqual(backend.stops, 1)

    def test_profiler_attached_plain_session_is_rejected_before_measurements(self):
        ledger, backend = self.run_fixture(Backend(profiler=True))
        self.assertFalse(ledger["completed"])
        self.assertFalse(backend.case_calls)
        self.assertFalse(ledger["eligible_for_unprofiled_timing"])
        self.assertEqual(backend.stops, 1)

    def test_instrumented_session_never_emits_unprofiled_eligibility(self):
        plan = s.default_plan(); plan["instrumentation"] = "nsys"
        ledger, backend = self.run_fixture(Backend(profiler=True), plan)
        self.assertTrue(ledger["completed"])
        self.assertTrue(all(r["instrumented"] for r in backend.case_calls))
        self.assertFalse(ledger["eligible_for_unprofiled_timing"])
        self.assertTrue(all(not e["eligible_for_unprofiled_timing"] for e in ledger["experiments"]))

    def test_plan_rejects_engine_mutation_variant_and_mixed_profile_policy(self):
        mutations = [lambda p: p["engine_arguments"].update(tensor_parallel_size=4),
                     lambda p: p["experiments"][0].update(variant="tokenspeed"),
                     lambda p: p["experiments"][0].update(instrumentation="nsys"),
                     lambda p: p["experiments"][1].update(id=p["experiments"][0]["id"]),
                     lambda p: p["experiments"][0]["cases"][0].update(length=512)]
        for change in mutations:
            plan = copy.deepcopy(s.default_plan()); change(plan)
            with self.assertRaises(ValueError):
                s.validate_plan(plan)

    def test_known_profiler_probe_does_not_export_environment_values(self):
        result = s.profiler_attachment_probe(environment={"NSYS_PROFILING_SESSION_ID": "fixture-value-never-exported"},
                                             maps_text="/opt/nsight/libToolsInjection64.so")
        self.assertTrue(result["known_profiler_detected"])
        self.assertNotIn("fixture-value-never-exported", json.dumps(result))
        plain = s.profiler_attachment_probe(environment={}, maps_text="libc.so")
        self.assertFalse(plain["known_profiler_detected"])
        tooling_only = s.profiler_attachment_probe(environment={}, maps_text="libcupti.so libnvperf.so")
        self.assertFalse(tooling_only["known_profiler_detected"])
        self.assertEqual(tooling_only["tooling_libraries_loaded"], ["libcupti", "libnvperf"])

    def test_existing_results_are_never_overwritten(self):
        self.out.mkdir(); (self.out/"original.json").write_text("{}")
        backend = Backend()
        with self.assertRaises(FileExistsError):
            self.run_fixture(backend)
        self.assertEqual(backend.starts, 0)

    def test_elapsed_deadline_stops_before_case_rpc_and_records_shutdown(self):
        plan = s.default_plan(); plan["max_session_seconds"] = 1
        with mock.patch.object(s.time, "monotonic", side_effect=[0, 2, 3]):
            ledger, backend = self.run_fixture(plan=plan)
        self.assertFalse(ledger["completed"])
        self.assertEqual(ledger["error"]["type"], "TimeoutError")
        self.assertFalse(backend.case_calls)
        self.assertEqual((backend.starts, backend.stops), (1, 1))


if __name__ == "__main__":
    unittest.main()
