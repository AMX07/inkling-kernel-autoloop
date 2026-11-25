"""Verify immutable relative-bias A/B evidence and compute predeclared gates.

Only unprofiled, complete, exact-correctness pairs support speed claims. Nsight
helper attribution is reported separately through launch correlation. No GPU
work and no original run mutations occur here.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import statistics
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.context_sweep.runtime import BASE_IMAGE, MODEL, REVISION, NATIVE_SOURCE_SHA256

try:
    from . import analyze_relative_bias_traces as traces
except ImportError:
    import analyze_relative_bias_traces as traces

MATRIX={(phase,length) for phase in ('prefill','decode') for length in (256,1024,4096)}
ATTENTION_SHA256='ff3164d9d02e1e317698e239d114796a091424f8df7bb1c310381127c7d4b1b1'
HELPER_SHA256='f0e68eece41a1844412a49085ef7d69375949facd7a252b140f8800e43414397'


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def require(value,message):
    if not value:raise ValueError(message)


def ranked(rows):
    require(isinstance(rows,list) and len(rows)==2 and all(isinstance(r,dict) and type(r.get('rank')) is int for r in rows)
            and {r['rank'] for r in rows}=={0,1},'Exactly two unique integer ranks required')
    return sorted(rows,key=lambda r:r['rank'])


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def plain_probe(row,instrumented):
    probe=row.get('profiler_probe',{})
    require(instrumented or (probe.get('maps_checked') is True and probe.get('known_profiler_detected') is False),
            'Unprofiled process attachment probe missing or failed')


def check_patch(value,*,restored=False,variant=None):
    require(value.get('target')=='vllm.models.inkling2.ops.attention._unshear_rel_logits'
            and value.get('source_sha256')==ATTENTION_SHA256 and value.get('original_helper_sha256')==HELPER_SHA256
            and value.get('installed') is (not restored) and value.get('restored') is restored
            and value.get('fallback_function_unchanged') is True and value.get('use_sheared_fa4') is False,
            'Reviewed helper patch/restoration or unchanged SM103 fallback not proven')
    if variant is not None:require(value.get('active_variant')==variant,'Helper dispatch variant mismatch')


def exact_helper_check(row,graph=False):
    fields=['passed','bitwise_equal','all_finite','output_contiguous','inputs_unchanged']
    if graph:fields+=['metadata_updated_in_place','native_graph_also_verified','input_addresses_unchanged']
    require(row.get('status')=='passed' and all(row.get(k) is True for k in fields)
            and sha(row.get('output_sha256')),'Exact helper correctness evidence incomplete')


def checkpoint_metadata(checkpoint):
    """Validate the pinned native snapshot schema, not a scalar map digest.

    StateCheckpoint.evidence emits metadata_sha256 as name -> SHA256. The
    frozen NativeSequenceState._record collects seven shared tensors plus
    four attention, four convolution, and two slot-map tensors per layer.
    The pinned 48-layer runtime and its source hashes are checked separately.
    """
    expected_regions={f'model.layers.{layer}.{suffix}':kind for layer in range(48)
                      for suffix,kind in (('attn','attention'),('conv_state','convolution'))}
    regions=checkpoint.get('regions',[])
    require(isinstance(regions,list) and len(regions)==96
            and all(isinstance(row,dict) for row in regions)
            and {row.get('name'):row.get('kind') for row in regions}==expected_regions,
            'Native checkpoint region coverage differs from all 48 attention/convolution layers')
    expected={'sequence.'+name for name in ('input_ids','positions','is_padding',
              'common.query_start_loc','common.query_start_loc_cpu','common.seq_lens','common.seq_lens_cpu')}
    for name,kind in expected_regions.items():
        fields=('query_start_loc','seq_lens','block_table','slot_mapping') if kind=='attention' else (
            'block_table','slot_mapping','seq_idx','query_start')
        expected.update('sequence.attention.'+name+'.'+field for field in fields)
        expected.add('sequence.slot_mapping.'+name)
    mapping=checkpoint.get('metadata_sha256')
    require(isinstance(mapping,dict) and bool(mapping) and set(mapping)==expected
            and all(isinstance(name,str) and bool(name) and sha(value) for name,value in mapping.items()),
            'Native metadata hash map must cover every named snapshot tensor with a valid SHA256')
    scalar=checkpoint.get('scalar_metadata_sha256')
    require(sha(scalar),'Native scalar metadata digest is invalid')
    return mapping,scalar


def verified_read(path,verified):
    require(path in verified,'Required evidence is not manifest-verified: '+path.name)
    return read(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    if path.is_symlink():raise ValueError('Symlinked evidence refused')
    return json.loads(path.read_text())


def safe_relative(name):
    return isinstance(name,str) and bool(name) and name!='.' and not Path(name).is_absolute() and '..' not in Path(name).parts


def verify_files(run):
    """Verify shared evidence and each session without conflating completeness.

    Root-manifest and frozen-source errors invalidate both sessions. A missing
    profile manifest/header only invalidates the profile session. Every file
    listed by the root manifest must still match, including captured tensors;
    no incomplete-download or damaged-root exception admits baseline results.
    """
    shared_errors=[];phase_errors={phase:[] for phase in ('baseline','profile')}
    errors=shared_errors;checks=0;verified=set()
    provenance=read(run/'source-provenance.json')
    if provenance.get('kind')!='relative-bias-source-provenance':errors.append('Unknown run provenance')
    for field,prefix in (('source_hashes',''),('baseline_source_hashes','baseline'),('baseline_evidence_hashes','baseline-evidence'),('contract_hashes','contracts')):
        mapping=provenance.get(field,{})
        if not mapping:errors.append('Missing frozen '+field)
        for name,expected in mapping.items():
            if not safe_relative(name):errors.append('Unsafe frozen source path');continue
            p=run/'executed-source'/prefix/name
            checks+=1
            if not p.is_file() or p.is_symlink() or digest(p)!=expected:errors.append('Frozen source mismatch: '+field+'/'+name)
    helper=provenance.get('original_helper',{})
    if safe_relative(helper.get('file')):
        p=run/'executed-source/original-helper'/helper['file'];checks+=1
        if not p.is_file() or p.is_symlink() or digest(p)!=helper.get('source_sha256') or helper.get('source_sha256')!=ATTENTION_SHA256:errors.append('Original attention source mismatch')
        if helper.get('function_source_sha256')!=HELPER_SHA256:errors.append('Original helper function mismatch')
    else:errors.append('Missing original helper provenance')
    plan_path=run/'executed-source/experiment-plan.json'
    if (not plan_path.is_file() or plan_path.is_symlink() or digest(plan_path)!=provenance.get('experiment_plan_file_sha256')
            or canonical_hash(read(plan_path))!=provenance.get('experiment_plan_sha256')):errors.append('Frozen experiment plan mismatch')
    actual_hashes={}
    manifest_entries={}
    for phase,manifest in ((None,run/'manifest.json'),('baseline',run/'benchmark-baseline/manifest.json'),('profile',run/'benchmark-profile/manifest.json')):
        errors=shared_errors if phase is None else phase_errors[phase]
        label=manifest.relative_to(run).as_posix()
        if not manifest.is_file():errors.append('Missing '+label);continue
        try:
            rows=read(manifest).get('files',[])
        except (ValueError,OSError,json.JSONDecodeError) as error:
            errors.append('Unreadable artifact manifest '+label+': '+str(error));continue
        if not isinstance(rows,list) or not rows:
            errors.append('Empty or invalid artifact manifest: '+label);continue
        seen={};manifest_entries[phase]=seen
        for row in rows:
            if not isinstance(row,dict):errors.append('Invalid manifest row: '+label);continue
            name=row.get('path')
            if not safe_relative(name) or name in seen:errors.append('Unsafe or duplicate manifest path: '+label);continue
            if type(row.get('bytes')) is not int or row['bytes']<0 or not sha(row.get('sha256')):
                errors.append('Invalid artifact size/hash: '+label+'/'+name);continue
            seen[name]=row
            p=manifest.parent/name;checks+=1
            relative=p.relative_to(run).as_posix()
            if not p.is_file() or p.is_symlink():errors.append('Missing artifact: '+relative);continue
            actual=actual_hashes.setdefault(p,None)
            if actual is None:actual=digest(p);actual_hashes[p]=actual
            if p.stat().st_size!=row.get('bytes') or actual!=row.get('sha256'):errors.append('Artifact hash/size mismatch: '+relative)
            else:verified.add(p)
    root_entries=manifest_entries.get(None,{})
    for phase in ('baseline','profile'):
        errors=phase_errors[phase]
        entries=manifest_entries.get(phase)
        if entries is not None:
            prefix='benchmark-'+phase+'/'
            expected={name.removeprefix(prefix):row for name,row in root_entries.items() if name.startswith(prefix)}
            if set(entries)!=set(expected):errors.append('Session/root manifest file sets differ: '+phase)
            for name,row in entries.items():
                root_row=expected.get(name,{})
                if any(row.get(k)!=root_row.get(k) for k in ('bytes','sha256')):
                    errors.append('Session/root manifest receipt mismatch: '+phase+'/'+name)
        for name in ('runtime.json','summary.json','config.json','inputs.json','runtime-inventory.json'):
            if run/('benchmark-'+phase)/name not in verified:errors.append('Unverified required header: '+phase+'/'+name)
    if run/'cloud-worker.json' not in verified:shared_errors.append('Unverified allocation worker receipt')
    all_errors=shared_errors+[error for phase in ('baseline','profile') for error in phase_errors[phase]]
    sessions={phase:{'verified':not shared_errors and not phase_errors[phase],
                     'errors':shared_errors+phase_errors[phase],
                     'scope':'Shared frozen sources, all root-manifest artifacts and this complete session manifest; every captured tensor remains required'}
              for phase in ('baseline','profile')}
    return {'verified':not all_errors,'checks':checks,'errors':all_errors,
            'shared':{'verified':not shared_errors,'errors':shared_errors},'sessions':sessions,
            'scope':'Global completeness remains false if either session is incomplete; individually verified baseline timings do not establish profile, adoption or serving eligibility'},verified


def stats(values):
    if not values or any(type(v) not in (float,int) or not math.isfinite(v) for v in values):raise ValueError('Nonfinite or empty observations')
    return {'count':len(values),'mean':statistics.mean(values),'sample_stddev':statistics.stdev(values) if len(values)>1 else 0,'median':statistics.median(values),'min':min(values),'max':max(values)}


def bootstrap(blocks, repetitions=5000):
    """Resample matched native/fused pairs within each confirmation block."""
    rng=random.Random(336);ratios=[]
    for _ in range(repetitions):
        samples=[pairs[rng.randrange(len(pairs))] for pairs in blocks for _ in pairs]
        ratios.append(sum(a for a,b in samples)/sum(b for a,b in samples))
    ratios.sort()
    return {'lower':ratios[int(.025*(repetitions-1))],'upper':ratios[int(.975*(repetitions-1))],
            'confidence':.95,'resamples':repetitions,'seed':336,'method':'Percentile paired bootstrap stratified by confirmation block; same sampled indices for native/fused. Within-session uncertainty only.'}


def check_session(run,phase,integrity,verified):
    instrumented=phase=='profile';folder=run/('benchmark-'+phase)
    result={'verified':False,'instrumented':instrumented,'errors':[]};context={}
    try:
        require(integrity['verified'],'Run artifact/source integrity failed')
        plan=read(run/'executed-source/experiment-plan.json');provenance=read(run/'source-provenance.json')
        require(plan.get('model')==MODEL and plan.get('revision')==REVISION and plan.get('base_image')==BASE_IMAGE
                and plan.get('gpu')=='B300:2' and plan.get('variants')==['native','fused']
                and plan.get('execution_mode')=='eager' and plan.get('confirmation_blocks')==2
                and plan.get('warmup_pairs_per_block')==5 and plan.get('measured_pairs_per_block')==10
                and len(plan.get('cases',[]))==6 and {(r['phase'],r['length']) for r in plan['cases']}==MATRIX,
                'Frozen experiment matrix/runtime differs from the reviewed protocol')
        require(provenance.get('model')==MODEL and provenance.get('model_revision')==REVISION and provenance.get('image')==BASE_IMAGE,
                'Source provenance model/image changed')
        summary=verified_read(folder/'summary.json',verified);runtime=verified_read(folder/'runtime.json',verified)
        config=verified_read(folder/'config.json',verified);inputs=verified_read(folder/'inputs.json',verified)
        inventory=verified_read(folder/'runtime-inventory.json',verified);allocation=verified_read(run/'cloud-worker.json',verified)
        result.update(summary_status=summary.get('status'),engine_start_count=summary.get('engine_start_count'),
                      startup_seconds=runtime.get('model_startup_seconds'),weights_retained_between_variants=runtime.get('weights_retained_between_variants'))
        require(summary.get('status')=='completed' and summary.get('completed') is True and summary.get('correctness_passed') is True
                and summary.get('instrumented') is instrumented and summary.get('engine_start_count')==1
                and summary.get('engine_shutdown_completed') is True and summary.get('weights_retained_between_variants') is True
                and summary.get('eligible_for_unprofiled_timing') is (not instrumented),'Completed resident session gates failed')
        require(runtime.get('model_loaded') is True and runtime.get('engine_start_count')==1
                and runtime.get('weights_retained_between_variants') is True,'One resident model load not proven')
        plan_hash=canonical_hash(plan)
        require(summary.get('plan_sha256')==plan_hash and config.get('plan_sha256')==plan_hash
                and canonical_hash({k:config.get(k) for k in plan})==plan_hash and config.get('instrumented') is instrumented
                and runtime.get('engine_arguments')==plan['engine_arguments'],'Runtime/config plan hashes or engine arguments differ')
        require(inventory.get('gpu_requested') is True and inventory.get('active_runtime',{}).get('torch_cuda')=='13.2',
                'Actual child GPU runtime inventory is missing or incompatible')
        ids=inputs.get('token_ids',[]);rng=random.Random(plan['seed']);expected_ids=[rng.randrange(200058) for _ in range(4097)]
        require(ids==expected_ids and inputs.get('seed')==plan['seed']
                and inputs.get('sha256')==hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                'Actual token IDs/hash differ from the seeded frozen input policy')
        rows=ranked(runtime.get('workers'));identity={};input_captures={}
        devices=allocation.get('allocation_gpu_identity',[])
        require(len(devices)==2 and all('B300' in d.get('name','') for d in devices),'Two B300 allocation devices not proven')
        normalize=lambda v:str(uuid.UUID(str(v).removeprefix('GPU-')))
        device_uuids={normalize(d['uuid']) for d in devices}
        require({int(d['index']) for d in devices}=={0,1},'Allocation device indices are not 0 and 1')
        require(len(device_uuids)==2 and {normalize(r['device_uuid']) for r in rows}==device_uuids,'Worker UUIDs do not match the allocation')
        candidate_sha=provenance['source_hashes']['experiments/relative_bias/kernel.py']
        for row in rows:
            rank=row['rank'];require(type(row.get('worker_pid')) is int and row['worker_pid']>0 and 'B300' in row.get('device_name',''),'Worker PID/device missing')
            identity[rank]=(row['worker_pid'],row['device_uuid']);plain_probe(row,instrumented);check_patch(row.get('helper_patch',{}),variant='native')
            native=row.get('native_before_patch_evidence',{});require(native.get('native_path_verified') is True and native.get('loaded_tokenspeed_module_names')==[],
                'Native pre-patch model/router evidence failed')
            files=native.get('installed_sources',{})
            require(set(files)==set(NATIVE_SOURCE_SHA256) and all(files[n].get('actual_sha256')==v and files[n].get('expected_sha256')==v
                    and files[n].get('matches_pinned_baseline') is True for n,v in NATIVE_SOURCE_SHA256.items()),'Installed native source hashes differ')
            checks=native.get('loaded_call_checks',{})
            require(all(checks.get(k) is True for k in ('loaded_model_is_native_class','router_alias_is_native_function','activation_alias_is_native_function'))
                    and checks.get('route_function_module')=='vllm.models.inkling2.ops.moe','Loaded native call aliases differ')
            correctness=row.get('adversarial_correctness',{})
            require(correctness==verified_read(folder/f'correctness-rank-{rank}.json',verified),'Worker/global correctness receipts differ')
            require(correctness.get('passed') is True and correctness.get('status')=='passed'
                    and correctness.get('native_source_sha256')==ATTENTION_SHA256 and correctness.get('candidate_source_sha256')==candidate_sha
                    and correctness.get('eager_case_count')==39 and len(correctness.get('eager_cases',[]))==39
                    and correctness.get('graph_checks_requested') is True and correctness.get('graph_case_count')==18
                    and len(correctness.get('graph_cases',[]))==18,'Adversarial/graph helper matrix incomplete')
            require(len({r.get('label') for r in correctness['eager_cases']})==39
                    and len({r.get('label') for r in correctness['graph_cases']})==18,'Adversarial/graph cases are duplicated')
            for item in correctness['eager_cases']:exact_helper_check(item)
            for item in correctness['graph_cases']:exact_helper_check(item,True)
            index=verified_read(folder/f'captured-inputs/rank-{rank}/index.json',verified);captures={}
            require(bool(index.get('captures')),'No actual prefill helper tensors captured')
            for capture in index['captures']:
                require(capture.get('rank')==rank and safe_relative(capture.get('file')),'Actual input rank/path invalid')
                artifact=folder/capture['file'];require(artifact in verified and artifact.suffix=='.pt'
                    and artifact.stat().st_size==capture.get('file_bytes') and digest(artifact)==capture.get('file_sha256'),
                    'Serialized actual helper tensor hash mismatch')
                key=capture.get('capture_id');require(sha(key) and key not in captures,'Duplicate/invalid actual input ID')
                captures[key]=capture
            input_captures[rank]=captures
        require(len({pid for pid,_ in identity.values()})==2,'Two distinct native worker processes not proven')
        for row in ranked(summary.get('patch_restoration')):
            require(row.get('worker_pid')==identity[row['rank']][0],'Worker restarted before helper restoration')
            check_patch(row.get('helper_patch',{}),restored=True)
        entries=summary.get('cases',[])
        require(summary.get('case_count')==6 and summary.get('completed_case_count')==6 and len(entries)==6
                and {(r.get('phase'),r.get('length')) for r in entries}==MATRIX
                and all(r.get('status')=='completed' and r.get('case_id')==f"{r['phase']}-L{r['length']}"
                        and r.get('result_file')==f"case-{r['phase']}-L{r['length']}.json" for r in entries),'Exact six-case matrix not completed')
        require({p.name for p in folder.glob('case-*.json')}=={e['result_file'] for e in entries},'Extra or missing case files')
        result['workers']=[{k:w.get(k) for k in ('rank','worker_pid','device_uuid','device_name','torch','torch_cuda','helper_patch')}
                           | {'adversarial_eager_cases':39,'graph_checks':18} for w in rows]
        result['verified']=True
        device_to_rank={int(d['index']):next(r['rank'] for r in rows if normalize(r['device_uuid'])==normalize(d['uuid'])) for d in devices}
        context={'identity':identity,'instrumented':instrumented,'plan':plan,'input_captures':input_captures,'folder':folder,'device_to_rank':device_to_rank}
    except (ValueError,KeyError,TypeError,OSError,json.JSONDecodeError) as error:
        result['errors'].append(str(error))
    return result,context


def check_microbenchmark(raw_rows,context,phase,length,verified,helper_checks):
    """Check the separate helper study; never promote its result to model latency."""
    output=copy.deepcopy(raw_rows);errors=[]
    try:
        rows=ranked(output);captured=ranked(helper_checks)
        for row in rows:
            rank=row['rank'];instrumented=context['instrumented']
            source=context['folder']/f'helper-microbench-{phase}-L{length}-rank-{rank}.json'
            require(row==verified_read(source,verified),'Helper microbenchmark differs from its verified standalone receipt')
            require(row.get('instrumented') is instrumented and row.get('execution_mode')=='eager','Helper instrumentation/mode differs')
            reports=row.get('reports',[])
            expected_ids={c['capture_id'] for c in captured[rank].get('captures',[])}
            require({r.get('capture_id') for r in reports}==expected_ids,'Helper timing input IDs do not match this case actual-input checks')
            if not reports:
                require(phase=='decode' and row.get('applicable') is False and row.get('status')=='not_applicable_zero_helper_calls',
                        'Helper timing absence is not an explicit decode control')
                row['analysis_verified']=True;continue
            require(row.get('applicable') is True and row.get('status')=='observed','Helper timing applicability missing')
            require(len({r.get('capture_id') for r in reports})==len(reports),'Duplicate helper microbenchmark capture')
            for report in reports:
                require(report.get('capture_id') in context['input_captures'][rank],'Helper timing uses an unverified actual input')
                rounds=report.get('warmup_rounds',[])
                require(1<=len(rounds)<=3,'Helper warmup round count invalid')
                for index,warmups in enumerate(rounds):
                    require(len(warmups)==10 and [(r.get('iteration'),r.get('variant')) for r in warmups]==
                            [(i,v) for i in range(5) for v in ('native','fused')]
                            and all(isinstance(r.get('compilation_events'),list) for r in warmups),'Helper warmup observations incomplete')
                    if index<len(rounds)-1:require(any(r['compilation_events'] for r in warmups),'Clean helper warmups were discarded')
                observations=report.get('observations',[])
                expected=[(i,v) for i in range(10) for v in (('native','fused') if i%2==0 else ('fused','native'))]
                require(len(observations)==20 and [(r.get('iteration'),r.get('variant')) for r in observations]==expected,
                        'Helper measured pairs incomplete or reordered')
                for observation in observations:
                    require(observation.get('repeats')==20 and positive(observation.get('wall_us_per_call'))
                            and positive(observation.get('cuda_event_us_per_call')) and isinstance(observation.get('compilation_events'),list),
                            'Helper repetitions/durations/compilation evidence invalid')
                    before=observation.get('before_allocated_bytes');peak=observation.get('peak_allocated_bytes');delta=observation.get('incremental_peak_allocated_bytes')
                    require(all(type(v) is int and v>=0 for v in (before,peak,delta)) and peak>=before and delta==peak-before,
                            'Helper incremental allocation counters inconsistent')
                clean=not any(r['compilation_events'] for r in rounds[-1]+observations)
                eligible=clean and not instrumented
                require(report.get('eligible_for_helper_timing') is eligible,'Helper timing eligibility disagrees with raw compilation/instrumentation')
                for variant in ('native','fused'):
                    for metric in ('wall_us_per_call','cuda_event_us_per_call'):
                        expected_stats=stats([r[metric] for r in observations if r['variant']==variant])
                        actual_stats=report.get('summary',{}).get(variant,{}).get(metric,{})
                        require(all(type(actual_stats.get(k)) in (int,float) and math.isclose(v,actual_stats[k],rel_tol=1e-12,abs_tol=1e-15)
                                    for k,v in expected_stats.items()),'Helper summary statistics disagree with retained observations')
                report['analysis_verified']=True
            row['analysis_verified']=True
    except (ValueError,KeyError,TypeError,OSError) as error:
        errors.append(str(error))
        for row in output:
            row['analysis_verified']=False
            for report in row.get('reports',[]):
                report['recorded_eligible_for_helper_timing']=report.get('eligible_for_helper_timing')
                report['eligible_for_helper_timing']=False;report['analysis_verified']=False
    return output,{'verified':not errors,'errors':errors,'scope':'Separate helper timings; compilation-tainted or instrumented observations remain diagnostic, never a model latency claim'}


def analyze_case(path,session_ok,verified,context=None):
    raw=read(path);phase=raw.get('phase');length=raw.get('length');instrumented=raw.get('instrumented')
    reasons=[];pairs=[];blocks=[];observations=defaultdict(list);references={};call_counts=defaultdict(list);invocations=0
    helper_checks=raw.get('captured_input_correctness',[]);context=context or {}

    def iteration(item,variant,stage,index,attempt,block,allow_compilation=False):
        nonlocal invocations
        rows=ranked(item['ranks']);compiled=False
        for row in rows:
            rank=row['rank'];require(row.get('status')=='completed' and row.get('variant')==variant
                and row.get('stage')==stage and row.get('iteration')==index and row.get('attempt')==attempt
                and row.get('block_index')==block,'Rank observation identity/order differs from the invocation')
            require(row.get('worker_pid')==context['identity'][rank][0],'Resident worker PID changed')
            check_patch(row.get('helper_patch',{}),variant=variant)
            require(positive(row.get('wall_seconds')) and positive(row.get('cuda_event_ms')),'Invalid rank timing duration')
            expected_capture=instrumented and stage=='measured'
            require(row.get('capture_active') is expected_capture,'Rank instrumentation mismatch')
            require(row.get('nvtx_range')==f'relative_bias/{phase}/L{length}/{variant}/{stage}/iteration-{index}/attempt-{attempt}/block-{block}',
                    'Rank NVTX identity differs from its recorded invocation')
            events=row.get('compilation_events');require(isinstance(events,list),'Missing rank compilation observer events')
            compiled|=bool(events);require(allow_compilation or not events,'Compilation contaminated an accepted warmup/measurement')
            calls=row.get('helper_calls');require(type(calls) is int and calls>=0 and (phase=='decode' or calls>0),'Prefill helper not exercised or invalid call count')
            call_counts[rank].append(calls)
            correct=row.get('correctness',{})
            require(all(correct.get(k) is True for k in ('passed','state_metadata_fixed','write_boundaries_verified'))
                    and correct.get('reference_variant')=='native' and type(correct.get('logits_present')) is bool
                    and correct.get('helper_applicable') is (calls>0),'Native reference/state correctness failed')
            present=correct['logits_present']
            if stage=='preparation' and variant=='native':
                references[rank]=(present,correct.get('reference_sha256'))
                require(correct.get('reference_established') is present,'Only native preparation establishes the reference')
            require(rank in references and references[rank][0] is present,'Output rank changed or native reference missing')
            if present:
                require(correct.get('finite') is True and correct.get('byte_exact') is True
                        and sha(correct.get('sha256')) and correct.get('sha256')==correct.get('reference_sha256')==references[rank][1],
                        'Full-model finite logits differ from the one native reference')
            if not (stage=='preparation' and variant=='native'):
                require(correct.get('reference_established') is False,'Native output reference was replaced')
            memory=row.get('memory',{});before=memory.get('before',{})
            for kind in ('allocated','reserved'):
                first=before.get(kind+'_bytes');peak=memory.get('peak_'+kind+'_bytes');after=memory.get('after_'+kind+'_bytes')
                delta=memory.get('incremental_peak_'+kind+'_bytes')
                require(all(type(v) is int and v>=0 for v in (first,peak,after,delta)) and peak>=max(first,after) and delta==peak-first,
                        'Allocator counter/delta evidence inconsistent')
        require(any(v[0] for v in references.values()),'No rank produced reference logits')
        require(item.get('correctness_passed') is True and item.get('compilation_observed') is compiled,'Combined correctness/compilation differs from ranks')
        maximum=max(r['wall_seconds'] for r in rows)
        require(positive(item.get('two_rank_completion_seconds')) and math.isclose(maximum,item['two_rank_completion_seconds'],rel_tol=1e-12),
                'Two-rank metric inconsistent with synchronized rank maxima')
        invocations+=1
        return maximum*1000,rows,compiled

    def pair(value,stage,index,attempt,block,allow_compilation=False,retain=False):
        expected=['native','fused'] if (index+block)%2==0 else ['fused','native']
        require(value.get('order')==expected and set(value.get('observations',{}))=={'native','fused'},'Pair order/variants differ from the frozen protocol')
        result={name:iteration(value['observations'][name],name,stage,index,attempt,block,allow_compilation) for name in expected}
        require(all(value.get(k) is True for k in ('correctness_passed','helper_counts_match','native_reference_matches','capture_state_matches')),
                'Pair correctness/capture gates failed')
        require(all(result['native'][1][rank]['helper_calls']==result['fused'][1][rank]['helper_calls'] for rank in (0,1)),
                'Actual native/fused helper call counts differ')
        compiled=any(v[2] for v in result.values());require(value.get('compilation_observed') is compiled,'Pair compilation summary differs from raw events')
        ratio=result['native'][0]/result['fused'][0]
        require(positive(value.get('native_over_fused_wall_ratio')) and math.isclose(value['native_over_fused_wall_ratio'],ratio,rel_tol=1e-12),'Pair ratio inconsistent')
        if retain:
            for name,(wall,rows,_) in result.items():
                observations[name].append({'block':block,'iteration':index,'wall_ms':wall,
                    'ranks':[{'rank':r['rank'],'worker_pid':r['worker_pid'],'wall_ms':r['wall_seconds']*1000,
                              'cuda_event_ms':r['cuda_event_ms'],'helper_calls':r['helper_calls'],'memory':r['memory']} for r in rows]})
        return (result['native'][0],result['fused'][0]),compiled

    try:
        require(path in verified,'Case file not manifest-verified');require(session_ok and context,'Owning session not eligible')
        require((phase,length) in MATRIX and raw.get('case_id')==f'{phase}-L{length}'
                and path.name==f'case-{phase}-L{length}.json' and raw.get('status')=='completed' and raw.get('completed') is True,
                'Case identity/completion invalid')
        require(instrumented is context['instrumented'] and raw.get('eligible_for_unprofiled_timing') is (not instrumented),'Case instrumentation/eligibility mismatch')
        require(raw.get('query_tokens')==(length if phase=='prefill' else 1) and raw.get('history_tokens')==(0 if phase=='prefill' else length),'Case token/history boundary changed')
        preparations=raw.get('preparation',[]);require(len(preparations)==2,'Exactly one native and one fused preparation required')
        for index,(variant,item) in enumerate(zip(('native','fused'),preparations)):
            iteration(item,variant,'preparation',index,0,0,True)
        raw_blocks=raw.get('blocks',[])
        require(len(raw_blocks)==2 and [b.get('block_index') for b in raw_blocks]==[0,1],'Two distinct ordered confirmation blocks required')
        for block in raw_blocks:
            block_index=block['block_index'];attempts=block.get('attempts',[]);selected=block.get('accepted_attempt')
            require(type(selected) is int and selected in (0,1) and len(attempts)==selected+1
                    and [a.get('attempt') for a in attempts]==list(range(len(attempts))) and block.get('completed') is True,
                    'Accepted attempt must be the unique final bounded attempt')
            for a in attempts:
                number=a['attempt'];accepted=number==selected;rounds=a.get('warmup_preparation_rounds',[])
                require(isinstance(rounds,list) and len(rounds)<=2 and all(len(r)==5 for r in rounds),'Retained warmup round count invalid')
                for round_index,values in enumerate(rounds):
                    dirty=False
                    for index,value in enumerate(values):dirty|=pair(value,'warmup',round_index*5+index,number,block_index,True)[1]
                    require(dirty,'Clean warmup set was selectively discarded')
                require(len(a.get('warmups',[]))==5 and len(a.get('measurements',[]))==10,'Five warmup pairs and ten measured pairs required')
                for index,value in enumerate(a['warmups']):pair(value,'warmup',len(rounds)*5+index,number,block_index)
                block_pairs=[];dirty=False
                for index,value in enumerate(a['measurements']):
                    values,compiled=pair(value,'measured',index,number,block_index,not accepted,accepted)
                    block_pairs.append(values);dirty|=compiled
                require(a.get('summary',{}).get('accepted') is accepted,'Recorded accepted-attempt summary disagrees')
                require((not dirty) if accepted else dirty,'Rejected measured attempt lacked observed compilation')
                capture=a.get('capture',[])
                require((len(capture)==2) if instrumented else capture==[],'Profiler start/stop receipt count differs')
                if instrumented:
                    for rows,enabled in zip(capture,(True,False)):
                        require(all(r.get('return_code')==0 and r.get('capture_active') is enabled for r in ranked(rows)),'Profiler start/stop failed on a rank')
                if accepted:
                    pairs.append(block_pairs)
                    blocks.append({'block':block_index,'accepted_attempt':selected,'native_ms':stats([a for a,b in block_pairs]),
                        'fused_ms':stats([b for a,b in block_pairs]),'speedup':sum(a for a,b in block_pairs)/sum(b for a,b in block_pairs)})
        for row in ranked(helper_checks):
            rank=row['rank'];all_zero=all(count==0 for count in call_counts[rank])
            if phase=='decode' and all_zero:
                require(row.get('applicable') is False and row.get('passed') is None and row.get('status')=='not_applicable'
                        and row.get('captures')==[] and row.get('checks')==[],'Zero-helper decode control receipt is not explicitly not applicable')
            else:
                require(row.get('applicable') is True and row.get('passed') is True and bool(row.get('captures')),'Actual-input helper correctness missing')
                captures={c['capture_id']:c for c in row['captures']};checks=row.get('checks',[])
                require(len(captures)==len(row['captures']) and len(checks)==len(captures)
                        and {c.get('capture_id') for c in checks}==set(captures),'Actual-input correctness coverage differs from captures')
                for capture_id,capture in captures.items():
                    require(capture==context['input_captures'][rank].get(capture_id),'Actual-input capture differs from the hash-verified index')
                for check in checks:exact_helper_check(check['result'])
        prepared=ranked(raw.get('prepared_workers'))
        for row in ranked(raw.get('final_state')):
            rank=row['rank'];require((row.get('worker_pid'),row.get('device_uuid'))==context['identity'][rank],'Resident worker/device changed')
            check_patch(row.get('helper_patch',{}));plain_probe(row,instrumented)
            require(row.get('helper_source_unchanged') is True,'Helper source changed during a case')
            state=row['state'];checkpoint=row['checkpoint'];initial=prepared[rank]
            require(state==initial.get('state') and initial.get('execution_mode')=='eager'
                    and initial.get('history_preparation_variant')=='native' and state.get('region_counts')=={'attention':48,'convolution':48}
                    and (state.get('query_tokens'),state.get('history_tokens'),state.get('total_sequence_tokens'))==
                    ((length,0,length) if phase=='prefill' else (1,length,length+1)),'Native state or history preparation boundary differs')
            require(all(checkpoint.get(k)==invocations for k in ('restore_count','verified_restore_count','verified_write_boundary_count')),
                    'Not every retained invocation has exact cache/state restore and write-boundary checks')
            require(checkpoint_metadata(checkpoint)==checkpoint_metadata(initial['checkpoint']),
                    'Native sequence metadata changed')
            require(all(row.get('compilation_monitor',{}).get(k) is True for k in ('native_jit_monitor_active','native_jit_callback','triton_autotuner_bench')),
                    'Required compilation observers not active')
    except (ValueError,KeyError,TypeError,IndexError,ZeroDivisionError) as error:
        reasons.append(str(error))
    variants={name:stats([r['wall_ms'] for r in rows]) for name,rows in observations.items() if rows}
    ratio=variants['native']['mean']/variants['fused']['mean'] if len(variants)==2 and not reasons else None
    helper_timing,helper_timing_proof=check_microbenchmark(raw.get('helper_microbenchmark',[]),context,phase,length,verified,helper_checks)
    return {'case_id':raw.get('case_id'),'phase':phase,'length':length,'instrumented':instrumented,
        'verified':not reasons,'eligible_for_timing':not reasons and instrumented is False,'reasons':sorted(set(reasons)),
        'latency_ms':variants,'speedup':ratio,'latency_reduction_percent':100*(1-1/ratio) if ratio else None,
        'paired_bootstrap95':bootstrap(pairs) if len(pairs)==2 and not reasons else None,'blocks':blocks,'observations':dict(observations),
        'verified_invocations_per_rank':invocations,'helper_microbenchmark':helper_timing,
        'helper_microbenchmark_verification':helper_timing_proof,'captured_input_correctness':helper_checks}


def analyze(run):
    integrity,verified=verify_files(run);receipt=read(run/'cloud-evidence.json');sessions={};cases={};contexts={}
    for phase,instrumented in (('baseline',False),('profile',True)):
        folder=run/('benchmark-'+phase)
        if not (folder/'summary.json').is_file():sessions[phase]={'verified':False};cases[phase]=[];continue
        session_integrity=integrity['sessions'][phase]
        sessions[phase],contexts[phase]=check_session(run,phase,session_integrity,verified)
        sessions[phase]['artifact_integrity']=session_integrity
        cases[phase]=[analyze_case(p,sessions[phase]['verified'],verified,contexts[phase]) for p in sorted(folder.glob('case-*.json'))]
    trace_rows=[];trace_files=[]
    for path in sorted((run/'profile').glob('*.sqlite')):
        entry={'file':path.relative_to(run).as_posix(),'verified':False}
        try:
            if path not in verified or path.with_suffix('.nsys-rep') not in verified:raise ValueError('Unverified trace pair')
            rows=traces.trace(path);trace_rows.extend(rows);entry.update(verified=True,ranges=len(rows),sha256=digest(path))
        except (ValueError,sqlite3.Error) as error:entry['error']=str(error)
        trace_files.append(entry)
    profile=[]
    for case in cases['profile']:
        accepted={b['block']:b['accepted_attempt'] for b in case['blocks']}
        matching=[r for r in trace_rows if r['phase']==case['phase'] and r['length']==case['length'] and accepted.get(r['block'])==r['attempt']]
        devices=[];errors=[]
        for device in (0,1):
            for variant in ('native','fused'):
                rows=[r for r in matching if r['device']==device and r['variant']==variant]
                identities={(r['block'],r['iteration']) for r in rows}
                if len(rows)!=20 or identities!={(b,i) for b in (0,1) for i in range(10)}:errors.append('Expected20 unique ranges/device/variant')
                context=contexts.get('profile',{});rank=context.get('device_to_rank',{}).get(device)
                expected={(r['block'],r['iteration']):r for r in case['observations'].get(variant,[])}
                for row in rows:
                    key=(row['block'],row['iteration']);observed=expected.get(key)
                    if rank is None or observed is None:
                        errors.append('Trace lacks a verified matching harness observation');continue
                    actual=next((r for r in observed['ranks'] if r['rank']==rank),{})
                    if row['pid']!=actual.get('worker_pid') or row['helper_calls']!=actual.get('helper_calls'):
                        errors.append('Trace process/helper call count differs from the actual rank observation')
                    if row.get('all_measured_kernels_launch_correlated') is not True:
                        errors.append('Not every measured kernel has a CUDA launch correlation')
                if rows:
                    devices.append({'device':device,'variant':variant,'ranges':len(rows),
                        **{field:stats([r[field] for r in rows]) for field in ('range_wall_ms','kernel_count','kernel_union_ms','kernel_coverage_fraction','no_kernel_gap_ms','helper_calls','helper_kernel_count','helper_cpu_range_total_ms','helper_kernel_sum_ms','helper_kernel_union_ms','helper_kernel_union_fraction','helper_projection_spans_sum_ms')},'raw_ranges':rows})
        profile.append({'case_id':case['case_id'],'phase':case['phase'],'length':case['length'],
            'verified':case['verified'] and not errors and bool(trace_files) and all(t['verified'] for t in trace_files),'errors':sorted(set(errors)),'devices':devices})
    baseline=cases['baseline'];complete=len(baseline)==6 and {(c['phase'],c['length']) for c in baseline}==MATRIX and all(c['eligible_for_timing'] for c in baseline)
    primary=next((c for c in baseline if c['phase']=='prefill' and c['length']==4096),None)
    primary_gain=bool(complete and primary and all(b['speedup']>1 for b in primary['blocks']) and primary['paired_bootstrap95']['lower']>1)
    decode_no_regression=bool(complete and all(c['speedup']>=1 for c in baseline if c['phase']=='decode'))
    profile_complete=len(profile)==6 and all(c['verified'] for c in profile)
    full_evidence=integrity['verified'] and complete and profile_complete
    eager_winner=full_evidence and primary_gain and decode_no_regression
    allocation_info={'verified':False,'elapsed_gpu_seconds':None,'estimated_gpu_usd':None,'gpu_usd_per_second':None,
        'billing_note':'Estimate uses the previously recorded two-B300 rate; actual Modal billing is authoritative.',
        'app_context_exited':receipt.get('app_context_exited') is True,'app_context_receipt_scope':'Local launcher receipt; separate from GPU worker cleanup checks',
        'children_overlap_verified_absent':False,'phases':{}}
    if run/'cloud-worker.json' in verified:
        worker=read(run/'cloud-worker.json');elapsed=worker.get('elapsed_gpu_seconds');cost=worker.get('estimated_gpu_usd');rate=receipt.get('gpu_usd_per_second')
        valid=positive(elapsed) and positive(rate) and type(cost) in (int,float) and math.isfinite(cost) and cost>=0 and math.isclose(cost,elapsed*rate,rel_tol=1e-9)
        allocation_info.update(verified=bool(valid),elapsed_gpu_seconds=elapsed if valid else None,
            estimated_gpu_usd=cost if valid else None,gpu_usd_per_second=rate if valid else None,
            children_overlap_verified_absent=worker.get('children_overlap_verified_absent') is True)
        for name,phase in worker.get('phases',{}).items():
            if name not in ('baseline','profile'):continue
            idle=phase.get('post_phase_gpu_idle_check',{})
            allocation_info['phases'][name]={'status':phase.get('status'),'instrumentation':phase.get('instrumentation'),
                'elapsed_seconds':phase.get('elapsed_seconds'),'exit_code':phase.get('exit_code'),'timed_out':phase.get('timed_out'),
                'process_group_cleanup_verified':phase.get('process_group_cleanup_verified') is True,
                'post_phase_gpu_idle_verified':idle.get('verified') is True and idle.get('remaining_compute_processes')==[],
                'remaining_live_process_group_count':len(phase.get('remaining_live_process_group_pids',[])),
                'scope':'Verified phase worker receipt; no raw process names, logs or environment metadata'}
    return {'kind':'relative-bias-analysis-v1','run_id':run.name,'generated_at':datetime.now(timezone.utc).isoformat(),
        'analysis_source_sha256':digest(Path(__file__)),
        'trace_analyzer_source_sha256':digest(Path(traces.__file__)),
        'analysis_scope':'Posthoc read-only derived analysis executed after capture; exact current analyzer hashes are separate from the sources frozen before the run. Per-session artifact validation preserves every global error and does not waive profile or adoption requirements.',
        'integrity':integrity,'status':'complete' if full_evidence else 'incomplete',
        'sessions':sessions,'baseline_complete':complete,'profile_complete':profile_complete,
        'cases':baseline,'profile_cases':profile,'trace_files':trace_files,
        'decision':{'primary_repeatable_gain':primary_gain,'decode_mean_no_regression':decode_no_regression,
            'baseline_timing_gates_passed':primary_gain and decode_no_regression,
            'eager_winner':eager_winner,'eligible_for_serving_validation':eager_winner,'serving_validated':False,'adopted':False,
            'reason':'Candidate still requires separate native CUDA-graph serving validation' if eager_winner else 'At least one predeclared eager adoption gate failed or remains incomplete; independently verified baseline timing does not waive full Nsight evidence'},
        'cloud_status':receipt.get('status'),
        'allocation':allocation_info,
        'limitations':['Within-session paired variation does not establish across-host or production-workload reproducibility.',
            'Helper CPU spans, projected GPU spans and kernel-union time are separate diagnostics; only unprofiled complete forward latency supports speed claims.',
            'Native single-token decode may bypass this helper and remains an unchanged regression control.',
            'Torch allocator peaks exclude external KV VMM; no exact all-allocator peak is claimed.',
            'One pretrained inference model; inference only, with no maximum-context result.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'planning/relative-bias-analysis.json');args=p.parse_args()
    run=args.run if args.run.is_absolute() else (args.run.resolve() if args.run.exists() else ROOT/'runs'/args.run)
    result=analyze(run);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'run_id':run.name,'status':result['status'],'baseline_complete':result['baseline_complete'],'profile_complete':result['profile_complete'],'decision':result['decision']}))


if __name__=='__main__':main()
