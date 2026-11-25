import importlib.util
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('relative_traces',Path(__file__).resolve().parents[1]/'scripts/analyze_relative_bias_traces.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class AttributionTests(unittest.TestCase):
    def make(self,path,*,wrong_process=False):
        db=sqlite3.connect(path)
        db.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);')
        tid=10*0x1000000+5;process=10*0x1000000
        db.execute('INSERT INTO StringIds VALUES(1,?)',('fused_kernel',))
        db.execute('INSERT INTO NVTX_EVENTS VALUES(0,1000,?,?,NULL)',(tid,'relative_bias/prefill/L256/fused/measured/iteration-0/attempt-0/block-0'))
        db.execute('INSERT INTO NVTX_EVENTS VALUES(100,200,?,?,NULL)',(tid,'relative_bias_helper/fused/prefill/L256/measured/iteration-0/attempt-0/block-0/call-1'))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(110,150,?,7)',(tid+(0x1000000 if wrong_process else 0),))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(300,500,0,?,7,1)',(process,))
        db.commit();db.close()

    def test_async_kernel_after_helper_cpu_span_is_attributed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.sqlite';self.make(p)
            rows=module.trace(p)
            self.assertEqual(rows[0]['helper_kernel_count'],1)
            self.assertEqual(rows[0]['helper_kernel_union_ms'],.0002)
            self.assertEqual(rows[0]['helper_cpu_range_total_ms'],.0001)

    def test_reused_correlation_other_process_cannot_supply_attribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.sqlite';self.make(p,wrong_process=True)
            with self.assertRaisesRegex(ValueError,'correlated'):module.trace(p)

    def test_kernel_overlap_counted_once(self):
        self.assertEqual(module.union_ns([(0,10),(5,15),(20,25)]),20)

    def test_missing_launch_correlation_rejects_partial_kernel_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.sqlite';self.make(p)
            with sqlite3.connect(p) as db:
                db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(510,530,0,?,8,1)',(10*0x1000000,))
            with self.assertRaisesRegex(ValueError,'lacks a correlated'):module.trace(p)

    def test_zero_helper_decode_keeps_total_kernel_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'trace.sqlite';self.make(p)
            with sqlite3.connect(p) as db:
                db.execute('DELETE FROM NVTX_EVENTS WHERE start=100')
                db.execute('UPDATE NVTX_EVENTS SET text=replace(text,\'prefill\',\'decode\')')
            rows=module.trace(p)
            self.assertEqual(rows[0]['helper_calls'],0)
            self.assertEqual(rows[0]['helper_kernel_count'],0)
            self.assertEqual(rows[0]['kernel_count'],1)
            self.assertTrue(rows[0]['all_measured_kernels_launch_correlated'])


if __name__=='__main__':unittest.main()
