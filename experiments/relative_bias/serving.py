"""Conditional native CUDA-graph serving validation, retaining one model.

This is an opt-in follow-up after the eager candidate passes its declared gate.
It uses the existing serving engine settings through native LLM.generate, not
HTTP; its fresh paired results must not be subtracted from historical HTTP
latencies. Graphs are recaptured for each variant, weights are never reloaded.
The actual pinned LLM/graph APIs must pass CPU inspection before GPU execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

from profiler.artifacts import manifest, write_json
from experiments.context_sweep import runtime
from experiments.context_sweep.resident_session import profiler_attachment_probe


def engine_arguments():
    return dict(model=runtime.MODEL, revision=runtime.REVISION,
        tokenizer_revision=runtime.REVISION, tensor_parallel_size=2,
        distributed_executor_backend='mp', max_model_len=4096,
        max_num_seqs=4, max_num_batched_tokens=512,
        compilation_config={'prefill_cudagraph_capture_token_sizes':[256,512],
                            'decode_cudagraph_capture_request_sizes':[1,2,4]},
        quantization='inkling2', hf_overrides={'quantization_config':None},
        limit_mm_per_prompt={'image':0}, gpu_memory_utilization=.90,
        enable_prefix_caching=False, speculative_config=None, seed=0,
        disable_log_stats=False)


def plan():
    """CPU-only exact protocol; the launcher freezes and hash-checks this."""
    return {'kind':'relative-bias-native-serving-validation-v1','engine_arguments':engine_arguments(),
        'variant_order':['native','fused','fused','native'],'warmups_per_block':5,'measurements_per_block':10,
        'max_warmup_rounds':3,'max_measured_attempts':2,'maximum_child_seconds':1800,
        'compilation_policy':'Require5clean warmups; retain and retry the entire10-sample measured block once after any observed JIT or graph capture. Never remove individual observations.',
        'graph_policy':'Preserve native breakable PIECEWISE prefill and FULL decode. Prove256-token breakable replay with captured graph segments and eager attention-helper calls, plus1-token FULL decode replay on both ranks. The helper itself executes eagerly between captured graph segments.',
        'prompt_tokens':256,'output_tokens':32,'temperature':0,'ignore_eos':True,
        'statistics_policy':'disable_log_stats=False matches the historical serve default. Native statistics collection/logging is enabled equally for both variants and its cost remains inside the same generation boundary.',
        'request_metrics_schema':{'type':'vllm.v1.metrics.stats.RequestStateStats','source_schema':'pinned-vllm-RequestStateStats-v1',
            'arrival_time':'Frontend wall-clock seconds; never subtract from engine monotonic timestamps.',
            'engine_timestamps':'queued_ts, scheduled_ts, first_token_ts and last_token_ts use engine monotonic seconds.',
            'ttft':'first_token_latency in seconds, computed by native frontend statistics.',
            'tpot':'(last_token_ts-first_token_ts)/(num_generation_tokens-1);32 generated tokens yield31 decode intervals.'},
        'measurement_boundary':'Blocking native LLM.generate includes scheduling, sampling and frontend processing; excludes tokenizer, graph capture, weight loading. No HTTP transport.',
        'scope':'Same engine limits and CUDA-graph capture shapes as historical serving, fresh A/B comparison; not identical HTTP request latency.',
        'retention':'One engine startup/final shutdown; recapture graphs between variants without reloading weights.'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def native_request_metrics(metrics):
    """Allowlist actual pinned v1 fields without mixing their clock domains."""
    metric_type=type(metrics).__module__+'.'+type(metrics).__qualname__
    expected_type='vllm.v1.metrics.stats.RequestStateStats'
    if metric_type!=expected_type:
        raise RuntimeError('Expected native RequestStateStats; statistics must be enabled for both variants')
    numeric=('arrival_time','queued_ts','scheduled_ts','first_token_ts','last_token_ts','first_token_latency')
    output={'metrics_type':metric_type,'source_schema':'pinned-vllm-RequestStateStats-v1'}
    for name in numeric:
        value=getattr(metrics,name,None)
        if type(value) not in (float,int) or not math.isfinite(value) or value<0:
            raise RuntimeError('Missing or invalid native request statistic: '+name)
        output[name]=value
    for name in ('num_generation_tokens','num_preemptions'):
        value=getattr(metrics,name,None)
        if type(value) is not int or value<0:
            raise RuntimeError('Missing or invalid native request counter: '+name)
        output[name]=value
    corrupted=getattr(metrics,'is_corrupted',None)
    if type(corrupted) is not bool:raise RuntimeError('Missing native is_corrupted flag')
    output['is_corrupted']=corrupted
    return output


def install_piecewise_observer(worker,breakable_runner):
    """Count completed native breakable replay loops without changing segments."""
    original_replay=breakable_runner._replay
    worker._relative_breakable_original=original_replay
    def counted_piecewise(entry,args,kwargs):
        # Pinned _replay() calls entry.capture.replay(), which invokes each
        # stored graph.replay or eager callable in sequence. Counting after
        # successful return proves that loop ran, not merely its dispatch.
        capture=entry.capture
        if capture is None:raise RuntimeError('Piecewise replay has no captured entry')
        result=original_replay(entry,args,kwargs)
        key=(int(entry.batch_descriptor.num_tokens),int(capture.num_graphs),int(capture.num_eager_breaks))
        worker._relative_piecewise_replays[key]+=1
        return result
    breakable_runner._replay=counted_piecewise


def worker_setup(worker, out):
    import torch
    from vllm.models.inkling2.ops import attention
    from .adapter import HelperPatch
    from .kernel import unshear_rel_logits
    from experiments.context_sweep.benchmark import CompileMonitor
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    native=runtime.native_source_evidence(loaded_model=worker.model_runner.model)
    if native['native_path_verified'] is not True:raise RuntimeError('Native path not verified')
    probe=profiler_attachment_probe()
    if not probe['maps_checked'] or probe['known_profiler_detected']:raise RuntimeError('Serving baseline must be unattached')
    runner=worker.model_runner;properties=torch.cuda.get_device_properties(worker.device)
    if 'B300' not in properties.name:raise RuntimeError('B300 required')
    if runner.vllm_config.model_config.enforce_eager:raise RuntimeError('Serving requires native graphs')
    patch=HelperPatch(attention,unshear_rel_logits,out=out,rank=worker.rank,instrumented=False)
    worker._relative_serving_patch=patch
    worker._relative_serving_monitor=CompileMonitor()
    worker._relative_replays=Counter()
    worker._relative_piecewise_replays=Counter()
    worker._relative_helper_capture=Counter()
    invoke=patch.wrapper
    def tracked_helper(sheared,num_tokens,cache_seqlens,cu_seqlens_q,rel_extent):
        # This query checks CUDA stream capture state; it never reads tensor
        # values or synchronizes. Tracking exists only in this serving study.
        capturing=torch.cuda.is_current_stream_capturing()
        breakable=BreakableCUDAGraphCapture.current() is not None
        worker._relative_helper_capture[(patch.variant,num_tokens,rel_extent,capturing,breakable)]+=1
        return invoke(sheared,num_tokens,cache_seqlens,cu_seqlens_q,rel_extent)
    patch.wrapper=tracked_helper
    attention._unshear_rel_logits=tracked_helper
    manager=runner.cudagraph_manager
    if manager is None:raise RuntimeError('Native graph manager absent')
    if not manager.use_breakable_cg or manager.breakable_cg_runner is None:
        raise RuntimeError('Existing serving recipe requires native breakable piecewise prefill')
    install_piecewise_observer(worker,manager.breakable_cg_runner)
    # CPU-only replay counters demonstrate the actual graph branch exercised;
    # they do not inject synchronization or inspect GPU data during generation.
    worker._relative_graph_methods={}
    for name in ('run_fullgraph','run_pw_graph'):
        original=getattr(manager,name)
        worker._relative_graph_methods[name]=original
        def counted(*args,_name=name,_original=original,**kwargs):
            if _name=='run_fullgraph':
                desc=args[0] if args else kwargs['desc']
                tokens,mode=int(desc.num_tokens),str(desc.cg_mode)
                requests=desc.num_reqs
            else:
                from vllm.forward_context import get_forward_context
                context=get_forward_context()
                descriptor=context.batch_descriptor
                tokens=int(descriptor.num_tokens) if descriptor is not None else -1
                mode=str(context.cudagraph_runtime_mode);requests=None
            result=_original(*args,**kwargs)
            worker._relative_replays[(_name,mode,tokens,requests)]+=1
            return result
        setattr(manager,name,counted)
    return {'rank':worker.rank,'worker_pid':os.getpid(),'device_uuid':str(properties.uuid),
        'device_name':properties.name,'native_path':native,'profiler_probe':probe,
        'graph_mode':str(manager.cudagraph_mode),'helper_patch':patch.receipt(),
        'compilation_monitor':worker._relative_serving_monitor.coverage}


def _helper_capture_rows(worker):
    return [{'variant':variant,'num_tokens':tokens,'rel_extent':extent,
             'stream_was_capturing':capturing,'breakable_capture_context_active':breakable,'calls':count}
            for (variant,tokens,extent,capturing,breakable),count in sorted(worker._relative_helper_capture.items())]


def _piecewise_rows(worker):
    return [{'num_tokens':tokens,'replays':count,'graph_segments_per_replay':graphs,
             'eager_breaks_per_replay':eager,'graph_segment_replays':count*graphs,
             'eager_segment_calls':count*eager,'evidence':'native_breakable_capture_replay_completed'}
            for (tokens,graphs,eager),count in sorted(worker._relative_piecewise_replays.items())]


def _replay_rows(worker):
    return [{'method':method,'mode':mode,'num_tokens':tokens,'num_reqs':requests,'count':count,
             'evidence':'direct_native_graph_replay' if method=='run_fullgraph' else 'native_piecewise_dispatch'}
            for (method,mode,tokens,requests),count in sorted(worker._relative_replays.items(),key=lambda item:str(item[0]))]


def worker_recapture(worker, variant):
    import torch
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.distributed import get_tp_group
    if variant not in ('native','fused'):raise ValueError('Unknown variant')
    patch=worker._relative_serving_patch;patch.assert_intact()
    runner=worker.model_runner;manager=runner.cudagraph_manager
    torch.cuda.synchronize(worker.device);get_tp_group().barrier()
    patch.variant=variant;patch.calls=0
    worker._relative_helper_capture.clear()
    worker._relative_serving_monitor.context={'stage':'graph_recapture','variant':variant}
    event_start=len(worker._relative_serving_monitor.events)
    manager.release_graphs()
    CUDAGraphWrapper.clear_all_graphs()
    BreakableCUDAGraphWrapper.clear_all_graphs()
    start=time.perf_counter()
    captured_bytes=runner.capture_model()
    torch.cuda.synchronize(worker.device);get_tp_group().barrier()
    if not manager._graphs_captured or not manager.graphs:raise RuntimeError('No serving graphs recaptured')
    captures=_helper_capture_rows(worker)
    if not any(r['variant']==variant and not r['stream_was_capturing']
               and r['breakable_capture_context_active'] and r['num_tokens']==256 for r in captures):
        raise RuntimeError('Requested helper variant was not observed in the eager attention break during256-token piecewise capture')
    piecewise=[]
    for entry in manager.breakable_cg_runner.entries.values():
        capture=entry.capture
        if capture is not None:
            piecewise.append({'num_tokens':int(entry.batch_descriptor.num_tokens),
                              'graph_segments':int(capture.num_graphs),'eager_breaks':int(capture.num_eager_breaks)})
    if not any(row['num_tokens']==256 and row['graph_segments']>0 and row['eager_breaks']>0 for row in piecewise):
        raise RuntimeError('Native256-token piecewise graph/eager segment capture is absent')
    worker._relative_replays.clear()
    return {'rank':worker.rank,'variant':variant,'worker_pid':os.getpid(),
        'device_uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
        'recapture_seconds':time.perf_counter()-start,'capture_memory_bytes':captured_bytes,
        'helper_calls_during_recapture':patch.calls,'graph_count':len(manager.graphs),
        'helper_execution_during_recapture':captures,
        'prefill_eager_helper_during_piecewise_capture_verified':True,
        'helper_itself_is_a_cuda_graph_node':False,'piecewise_captures':piecewise,
        'recapture_compilation_events':worker._relative_serving_monitor.events[event_start:],
        'captured_token_counts':manager.captured_token_counts(),
        'helper_patch':patch.receipt(),'model_weights_reloaded':False}


def worker_begin_observation(worker,stage,iteration,attempt,block):
    import torch
    from vllm.compilation.counter import compilation_counter
    torch.cuda.synchronize(worker.device)
    patch=worker._relative_serving_patch;patch.assert_intact()
    monitor=worker._relative_serving_monitor
    monitor.context={'stage':stage,'iteration':iteration,'attempt':attempt,'block':block,'variant':patch.variant}
    worker._relative_observation_marker=len(monitor.events)
    worker._relative_graph_capture_before=int(compilation_counter.num_cudagraph_captured)
    worker._relative_replays.clear();worker._relative_piecewise_replays.clear();worker._relative_helper_capture.clear()
    return {'rank':worker.rank,'started':True}


def worker_receipt(worker, reset=False):
    import torch
    from vllm.compilation.counter import compilation_counter
    torch.cuda.synchronize(worker.device)
    patch=worker._relative_serving_patch;patch.assert_intact()
    value={'rank':worker.rank,'worker_pid':os.getpid(),
        'device_uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
        'graph_replays':_replay_rows(worker),'helper_patch':patch.receipt(),
        'piecewise_replays':_piecewise_rows(worker),
        'helper_execution':_helper_capture_rows(worker),
        'compilation_events':worker._relative_serving_monitor.events[worker._relative_observation_marker:],
        'new_graph_captures':int(compilation_counter.num_cudagraph_captured)-worker._relative_graph_capture_before}
    if reset:worker._relative_replays.clear();worker._relative_piecewise_replays.clear();worker._relative_helper_capture.clear()
    return value


def worker_restore(worker):
    import torch
    patch=worker._relative_serving_patch
    receipt=patch.restore()
    for name,method in worker._relative_graph_methods.items():
        setattr(worker.model_runner.cudagraph_manager,name,method)
    worker.model_runner.cudagraph_manager.breakable_cg_runner._replay=worker._relative_breakable_original
    return {'rank':worker.rank,'worker_pid':os.getpid(),
            'device_uuid':str(torch.cuda.get_device_properties(worker.device).uuid),
            'helper_patch':receipt,'graph_methods_restored':True}


def observation_graph_proof(workers,prompt_tokens=256):
    """Fail closed if the only proven graph replays were decode graphs."""
    if len(workers)!=2 or {r['rank'] for r in workers}!={0,1}:return False
    for worker in workers:
        replay=worker['graph_replays']
        if not any(r['num_tokens']==prompt_tokens and r['replays']>0
                   and r['graph_segments_per_replay']>0 and r['eager_breaks_per_replay']>0
                   for r in worker.get('piecewise_replays',[])):
            return False
        if not any(r['method']=='run_fullgraph' and r['num_tokens']==1 and r['count']>0 for r in replay):
            return False
        variant=worker.get('helper_patch',{}).get('active_variant')
        if not any(r['variant']==variant and r['num_tokens']==prompt_tokens and r['calls']>0
                   and not r['stream_was_capturing'] and not r.get('breakable_capture_context_active',False)
                   for r in worker['helper_execution']):
            return False
    return True


def observation_compiled(workers):
    return any(r['compilation_events'] or r['new_graph_captures']
               or any(h['stream_was_capturing'] or h.get('breakable_capture_context_active',False)
                      for h in r['helper_execution']) for r in workers)


def run(out,expected_plan=None):
    config=plan()
    if expected_plan is not None and canonical_hash(expected_plan)!=canonical_hash(config):
        raise ValueError('Frozen serving plan differs from this executable protocol')
    from vllm import LLM, SamplingParams
    from experiments.context_sweep.inventory import runtime_inventory
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',config)
    write_json(out/'runtime-inventory.json',runtime_inventory(phase='relative-bias-serving',gpu_requested=True))
    summary={'kind':'relative-bias-native-serving-validation-v1','status':'starting','completed':False,
             'engine_start_count':0,'blocks':[],'all_generated_tokens_equal':True,'plan_sha256':canonical_hash(config)}
    write_json(out/'summary.json',summary);llm=None
    def save():write_json(out/'summary.json',summary)
    def call(fn,*args):return llm.collective_rpc(runtime._invoke_with_config,timeout=600,args=(fn,args,{}))
    try:
        start=time.perf_counter();llm=LLM(**engine_arguments())
        summary.update(engine_start_count=1,model_loaded=True,model_startup_seconds=time.perf_counter()-start)
        identities=call(worker_setup,str(out))
        identity=lambda rows:sorted((r['rank'],r['worker_pid'],r['device_uuid']) for r in rows)
        expected_identity=identity(identities)
        if len(identities)!=2 or len({r['device_uuid'] for r in identities})!=2:raise RuntimeError('Expected two distinct B300 workers')
        write_json(out/'runtime.json',{'workers':identities,'engine_arguments':engine_arguments(),
            'model_loaded':True,'engine_start_count':1,'weights_retained_between_variants':True,
            'plan_sha256':canonical_hash(config)})
        tokenizer=llm.get_tokenizer()
        text='A careful experiment measures time and memory on a graphics processor. '
        token_ids=tokenizer.encode(text*max(8,256//6))[:256]
        if len(token_ids)!=256:raise RuntimeError('Prompt is not exactly256 tokens')
        write_json(out/'inputs.json',{'token_ids':token_ids,'sha256':hashlib.sha256(json.dumps(token_ids,separators=(',',':')).encode()).hexdigest(),'generator':'Pinned native tokenizer; historical serving benchmark text repeated then truncated to256 IDs'})
        sampling=SamplingParams(temperature=0,max_tokens=32,ignore_eos=True,seed=0)
        reference=None

        def observe(stage,iteration,attempt,block):
            nonlocal reference
            call(worker_begin_observation,stage,iteration,attempt,block)
            start=time.perf_counter()
            outputs=llm.generate([{'prompt_token_ids':token_ids}],sampling,use_tqdm=False)
            elapsed=time.perf_counter()-start
            workers=call(worker_receipt)
            if identity(workers)!=expected_identity:raise RuntimeError('Resident workers changed')
            if len(outputs)!=1 or len(outputs[0].outputs)!=1:raise RuntimeError('Expected one completed generation')
            result=outputs[0];generated=list(result.outputs[0].token_ids)
            if not result.finished or len(generated)!=32 or list(result.prompt_token_ids)!=token_ids:
                raise RuntimeError('Generation token-count/input mismatch')
            if reference is None:
                if config['variant_order'][block]!='native':raise RuntimeError('Only native output may establish the reference')
                reference=generated;write_json(out/'native-reference-output.json',{'token_ids':reference})
            exact=generated==reference
            if not exact:summary['all_generated_tokens_equal']=False
            safe_metrics=native_request_metrics(result.metrics)
            observation={'iteration':iteration,'wall_ms':elapsed*1000,'output_tokens':32,
                'output_sha256':hashlib.sha256(json.dumps(generated).encode()).hexdigest(),'exact_native_output':exact,
                'native_request_timestamps':safe_metrics,'workers':workers,
                'native_metrics_usable':safe_metrics['num_generation_tokens']==32
                    and safe_metrics['num_preemptions']==0 and not safe_metrics['is_corrupted'],
                'compilation_observed':observation_compiled(workers),
                'prefill_and_decode_graph_replay_verified':observation_graph_proof(workers)}
            return observation

        for block,variant in enumerate(config['variant_order']):
            recapture=call(worker_recapture,variant)
            if identity(recapture)!=expected_identity:raise RuntimeError('Worker identity changed during recapture')
            entry={'block':block,'variant':variant,'recapture':recapture,'attempts':[],
                   'accepted_attempt':None,'warmups':[],'measurements':[],'status':'running'}
            summary['blocks'].append(entry);save()
            for attempt_index in range(config['max_measured_attempts']):
                attempt={'attempt':attempt_index,'warmup_preparation_rounds':[],
                         'warmups':[],'measurements':[],'accepted':False}
                entry['attempts'].append(attempt)
                for round_index in range(config['max_warmup_rounds']):
                    for index in range(5):
                        observation=observe('warmup',round_index*5+index,attempt_index,block)
                        attempt['warmups'].append(observation);save()
                        if not observation['exact_native_output']:raise RuntimeError('Candidate/native generated tokens differ')
                        if not observation['native_metrics_usable']:raise RuntimeError('Native serving statistics record preemption, corrupted logits or a mismatched token count')
                    if not any(row['compilation_observed'] for row in attempt['warmups']):break
                    if round_index+1<config['max_warmup_rounds']:
                        attempt['warmup_preparation_rounds'].append(attempt['warmups']);attempt['warmups']=[]
                if any(row['compilation_observed'] for row in attempt['warmups']):
                    raise RuntimeError('Serving compilation persisted through three full warmup rounds')
                for index in range(10):
                    observation=observe('measured',index,attempt_index,block)
                    attempt['measurements'].append(observation);save()
                    if not observation['exact_native_output']:raise RuntimeError('Candidate/native generated tokens differ')
                    if not observation['native_metrics_usable']:raise RuntimeError('Native serving statistics record preemption, corrupted logits or a mismatched token count')
                compiled=any(row['compilation_observed'] for row in attempt['measurements'])
                graphs=all(row['prefill_and_decode_graph_replay_verified'] for row in attempt['warmups']+attempt['measurements'])
                attempt.update(compilation_observed=compiled,graph_replay_verified=graphs,accepted=not compiled and graphs)
                save()
                if attempt['accepted']:
                    entry.update(accepted_attempt=attempt_index,warmups=attempt['warmups'],measurements=attempt['measurements'])
                    break
                if not graphs:raise RuntimeError('Native piecewise prefill/eager helper and full-graph decode replays were not all proven')
            if entry['accepted_attempt'] is None:
                raise RuntimeError('Serving measured blocks remained compilation-contaminated after the retained retry')
            entry.update(status='completed');save()
        summary.update(status='completed',completed=True,weights_retained_between_variants=True)
    except Exception as error:
        summary.update(status='failed',error={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        if llm is not None:
            try:
                restored=call(worker_restore)
                summary['patch_restoration']=restored
                if not all(r['helper_patch']['restored'] and r['graph_methods_restored'] for r in restored):
                    raise RuntimeError('Original helper/graph-method restoration was not proven')
            except Exception as error:
                summary.update(status='failed',completed=False,restoration_error=str(error))
            try:
                # The pinned API is inspected before this code is authorized to run.
                llm.llm_engine.engine_core.shutdown()
                summary['engine_shutdown_completed']=True
            except Exception as error:
                summary.update(status='failed',completed=False,engine_shutdown_completed=False,shutdown_error=str(error))
        try:
            save()
        finally:
            manifest(out,kind='relative-bias-native-serving',status=summary['status'])
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--output-dir',type=Path)
    p.add_argument('--plan',type=Path)
    args=p.parse_args()
    if not args.run:
        print(json.dumps({'status':'plan_only','gpu_requested':False,'plan':plan(),'plan_sha256':canonical_hash(plan())}));return
    if args.output_dir is None:p.error('--output-dir is required')
    result=run(args.output_dir,expected_plan=json.loads(args.plan.read_text()) if args.plan else None)
    if result['completed'] is not True:raise SystemExit(1)


if __name__=='__main__':main()
