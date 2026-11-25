import unittest
import copy
import json
import hashlib
import random
from pathlib import Path
import tempfile
from unittest import mock
from scripts import analyze_relative_bias as a
from scripts.analyze_relative_bias import bootstrap,stats


def patch(variant='native'):
    return {'target':'vllm.models.inkling2.ops.attention._unshear_rel_logits','source_sha256':a.ATTENTION_SHA256,
            'original_helper_sha256':a.HELPER_SHA256,'installed':True,'restored':False,
            'fallback_function_unchanged':True,'use_sheared_fa4':False,'active_variant':variant}


def case_fixture(phase='prefill',instrumented=False):
    length=256;identity={rank:(100+rank,f'fixture-uuid-{rank}') for rank in (0,1)}
    calls=48 if phase=='prefill' else 0
    state={'query_tokens':length if phase=='prefill' else 1,'history_tokens':0 if phase=='prefill' else length,
           'total_sequence_tokens':length if phase=='prefill' else length+1,'region_counts':{'attention':48,'convolution':48}}
    regions=[{'name':f'model.layers.{layer}.{suffix}','kind':kind} for layer in range(48)
             for suffix,kind in (('attn','attention'),('conv_state','convolution'))]
    metadata={'sequence.'+name:'a'*64 for name in ('input_ids','positions','is_padding',
              'common.query_start_loc','common.query_start_loc_cpu','common.seq_lens','common.seq_lens_cpu')}
    for region in regions:
        fields=('query_start_loc','seq_lens','block_table','slot_mapping') if region['kind']=='attention' else (
            'block_table','slot_mapping','seq_idx','query_start')
        metadata.update({'sequence.attention.'+region['name']+'.'+field:'a'*64 for field in fields})
        metadata['sequence.slot_mapping.'+region['name']]='a'*64
    checkpoint={'metadata_sha256':metadata,'scalar_metadata_sha256':'b'*64,'regions':regions}
    context={'identity':identity,'instrumented':instrumented,'input_captures':{0:{},1:{}}}
    def iteration(variant,stage,index,attempt,block):
        wall=.002 if variant=='native' else .0016
        rows=[]
        for rank in (0,1):
            rows.append({'rank':rank,'status':'completed','variant':variant,'stage':stage,'iteration':index,'attempt':attempt,'block_index':block,
                'worker_pid':100+rank,'helper_patch':patch(variant),'wall_seconds':wall,'cuda_event_ms':.5,
                'capture_active':instrumented and stage=='measured','compilation_events':[],'helper_calls':calls,
                'nvtx_range':f'relative_bias/{phase}/L{length}/{variant}/{stage}/iteration-{index}/attempt-{attempt}/block-{block}',
                'correctness':{'passed':True,'state_metadata_fixed':True,'write_boundaries_verified':True,'reference_variant':'native',
                    'logits_present':True,'helper_applicable':bool(calls),'reference_established':stage=='preparation' and variant=='native',
                    'finite':True,'byte_exact':True,'sha256':'c'*64,'reference_sha256':'c'*64},
                'memory':{'before':{'allocated_bytes':100,'reserved_bytes':200},'peak_allocated_bytes':120,'peak_reserved_bytes':200,
                          'after_allocated_bytes':110,'after_reserved_bytes':200,'incremental_peak_allocated_bytes':20,'incremental_peak_reserved_bytes':0}})
        return {'ranks':rows,'correctness_passed':True,'compilation_observed':False,'two_rank_completion_seconds':wall,'controller_rpc_seconds':wall+.0001}
    def pair(stage,index,attempt,block):
        return {'order':['native','fused'] if (index+block)%2==0 else ['fused','native'],
            'observations':{v:iteration(v,stage,index,attempt,block) for v in ('native','fused')},
            'correctness_passed':True,'helper_counts_match':True,'native_reference_matches':True,'capture_state_matches':True,
            'compilation_observed':False,'native_over_fused_wall_ratio':1.25}
    raw={'case_id':f'{phase}-L{length}','phase':phase,'length':length,'instrumented':instrumented,'status':'completed','completed':True,
         'eligible_for_unprofiled_timing':not instrumented,'query_tokens':state['query_tokens'],'history_tokens':state['history_tokens'],
         'preparation':[iteration('native','preparation',0,0,0),iteration('fused','preparation',1,0,0)],'blocks':[],
         'captured_input_correctness':[],'prepared_workers':[],'final_state':[]}
    for block in (0,1):
        capture=[[{'rank':r,'return_code':0,'capture_active':active} for r in (0,1)] for active in (True,False)] if instrumented else []
        raw['blocks'].append({'block_index':block,'accepted_attempt':0,'completed':True,'attempts':[{'attempt':0,'warmup_preparation_rounds':[],
            'warmups':[pair('warmup',i,0,block) for i in range(5)],'measurements':[pair('measured',i,0,block) for i in range(10)],
            'capture':capture,'summary':{'accepted':True}}]})
    for rank in (0,1):
        raw['prepared_workers'].append({'rank':rank,'state':state,'checkpoint':copy.deepcopy(checkpoint),'execution_mode':'eager','history_preparation_variant':'native'})
        raw['final_state'].append({'rank':rank,'worker_pid':100+rank,'device_uuid':identity[rank][1],'helper_patch':patch('fused'),
            'helper_source_unchanged':True,'profiler_probe':{'maps_checked':True,'known_profiler_detected':False},'state':state,
            'checkpoint':{**copy.deepcopy(checkpoint),'restore_count':62,'verified_restore_count':62,'verified_write_boundary_count':62},
            'compilation_monitor':{k:True for k in ('native_jit_monitor_active','native_jit_callback','triton_autotuner_bench')}})
        if not calls:
            raw['captured_input_correctness'].append({'rank':rank,'applicable':False,'passed':None,'status':'not_applicable','captures':[],'checks':[]})
        else:
            capture={'rank':rank,'capture_id':str(rank)*64,'file':f'captured-inputs/rank-{rank}/capture.pt'}
            context['input_captures'][rank][capture['capture_id']]=capture
            raw['captured_input_correctness'].append({'rank':rank,'applicable':True,'passed':True,'captures':[capture],
                'checks':[{'capture_id':capture['capture_id'],'result':{'status':'passed','passed':True,'bitwise_equal':True,'all_finite':True,
                    'output_contiguous':True,'inputs_unchanged':True,'output_sha256':'d'*64}}]})
    return raw,context


def session_fixture(root):
    from experiments.relative_bias.benchmark import plan as experiment_plan
    plan=experiment_plan();folder=root/'benchmark-baseline';folder.mkdir()
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    write(root/'executed-source/experiment-plan.json',plan)
    candidate_sha='f'*64
    write(root/'source-provenance.json',{'model':a.MODEL,'model_revision':a.REVISION,'image':a.BASE_IMAGE,
        'source_hashes':{'experiments/relative_bias/kernel.py':candidate_sha}})
    identities=[f'00000000-0000-0000-0000-{rank+1:012d}' for rank in (0,1)]
    write(root/'cloud-worker.json',{'allocation_gpu_identity':[{'index':str(rank),'name':'NVIDIA B300','uuid':identities[rank]} for rank in (0,1)]})
    workers=[]
    for rank in (0,1):
        exact={'status':'passed','passed':True,'bitwise_equal':True,'all_finite':True,'output_contiguous':True,
               'inputs_unchanged':True,'output_sha256':'e'*64}
        correctness={'passed':True,'status':'passed','native_source_sha256':a.ATTENTION_SHA256,'candidate_source_sha256':candidate_sha,
            'eager_case_count':39,'eager_cases':[{**exact,'label':f'eager-{i}'} for i in range(39)],
            'graph_checks_requested':True,'graph_case_count':18,
            'graph_cases':[{**exact,'label':f'graph-{i}','metadata_updated_in_place':True,'native_graph_also_verified':True,
                            'input_addresses_unchanged':True} for i in range(18)]}
        native={'native_path_verified':True,'loaded_tokenspeed_module_names':[],
            'installed_sources':{name:{'actual_sha256':sha,'expected_sha256':sha,'matches_pinned_baseline':True}
                                 for name,sha in a.NATIVE_SOURCE_SHA256.items()},
            'loaded_call_checks':{'loaded_model_is_native_class':True,'router_alias_is_native_function':True,
                                 'activation_alias_is_native_function':True,'route_function_module':'vllm.models.inkling2.ops.moe'}}
        workers.append({'rank':rank,'device_name':'NVIDIA B300','worker_pid':100+rank,'device_uuid':identities[rank],
            'native_before_patch_evidence':native,'helper_patch':patch(),'adversarial_correctness':correctness,
            'profiler_probe':{'maps_checked':True,'known_profiler_detected':False}})
        write(folder/f'correctness-rank-{rank}.json',correctness)
        path=folder/f'captured-inputs/rank-{rank}/capture.pt';path.parent.mkdir(parents=True);path.write_bytes(b'CPU fixture bytes only')
        write(path.parent/'index.json',{'captures':[{'rank':rank,'capture_id':str(rank)*64,'file':str(path.relative_to(folder)),
            'file_bytes':path.stat().st_size,'file_sha256':a.digest(path)}]})
    entries=[]
    for phase,length in sorted(a.MATRIX):
        entry={'phase':phase,'length':length,'case_id':f'{phase}-L{length}','status':'completed','result_file':f'case-{phase}-L{length}.json'}
        entries.append(entry);write(folder/entry['result_file'],{})
    write(folder/'runtime.json',{'model_loaded':True,'engine_start_count':1,'weights_retained_between_variants':True,
        'model_startup_seconds':100,'engine_arguments':plan['engine_arguments'],'workers':workers})
    write(folder/'summary.json',{'status':'completed','completed':True,'correctness_passed':True,'instrumented':False,'engine_start_count':1,
        'engine_shutdown_completed':True,'weights_retained_between_variants':True,'eligible_for_unprofiled_timing':True,
        'plan_sha256':a.canonical_hash(plan),'case_count':6,'completed_case_count':6,'cases':entries,
        'patch_restoration':[{'rank':r,'worker_pid':100+r,'helper_patch':{**patch(),'installed':False,'restored':True}} for r in (0,1)]})
    write(folder/'config.json',{**plan,'plan_sha256':a.canonical_hash(plan),'instrumented':False})
    write(folder/'runtime-inventory.json',{'gpu_requested':True,'active_runtime':{'torch_cuda':'13.2'}})
    rng=random.Random(plan['seed']);ids=[rng.randrange(200058) for _ in range(4097)]
    write(folder/'inputs.json',{'token_ids':ids,'seed':plan['seed'],'sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()})
    return {p for p in root.rglob('*') if p.is_file()}


def manifested_baseline_fixture(root):
    """A complete hash-covered synthetic baseline with no profile manifest."""
    session_fixture(root)
    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    source_groups={}
    for field,prefix,name in (('source_hashes','','experiments/relative_bias/kernel.py'),
                             ('baseline_source_hashes','baseline','runtime.py'),
                             ('baseline_evidence_hashes','baseline-evidence','evidence.json'),
                             ('contract_hashes','contracts','protocol.json')):
        path=root/'executed-source'/prefix/name
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'synthetic CPU integrity fixture\n')
        source_groups[field]={name:a.digest(path)}
    helper_name='vllm/models/inkling2/ops/attention.py'
    helper=root/'executed-source/original-helper'/helper_name;helper.parent.mkdir(parents=True,exist_ok=True)
    helper.write_bytes((a.ROOT/'planning/tokenspeed-baseline-source'/helper_name).read_bytes())
    plan_path=root/'executed-source/experiment-plan.json'
    provenance=json.loads((root/'source-provenance.json').read_text())
    provenance.update(kind='relative-bias-source-provenance',**source_groups,
                      original_helper={'file':helper_name,'source_sha256':a.digest(helper),'function_source_sha256':a.HELPER_SHA256},
                      experiment_plan_file_sha256=a.digest(plan_path),experiment_plan_sha256=a.canonical_hash(json.loads(plan_path.read_text())))
    write(root/'source-provenance.json',provenance)
    candidate_sha=source_groups['source_hashes']['experiments/relative_bias/kernel.py']
    runtime=json.loads((root/'benchmark-baseline/runtime.json').read_text())
    for worker in runtime['workers']:
        worker['adversarial_correctness']['candidate_source_sha256']=candidate_sha
        write(root/f"benchmark-baseline/correctness-rank-{worker['rank']}.json",worker['adversarial_correctness'])
    write(root/'benchmark-baseline/runtime.json',runtime)
    write(root/'benchmark-baseline/worker-observations/warmup.json',{'status':'completed'})
    write(root/'cloud-evidence.json',{'status':'failed','gpu_usd_per_second':.003944})
    def manifest(folder):
        rows=[{'path':p.relative_to(folder).as_posix(),'bytes':p.stat().st_size,'sha256':a.digest(p)}
              for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='manifest.json']
        write(folder/'manifest.json',{'files':rows})
    manifest(root/'benchmark-baseline');manifest(root)


class AnalysisTests(unittest.TestCase):
    def analyze_fixture(self,raw,context):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/f"case-{raw['phase']}-L{raw['length']}.json";path.write_text(json.dumps(raw))
            return a.analyze_case(path,True,{path},context)

    def test_matched_pair_bootstrap_retains_covariance(self):
        blocks=[[(n*2.,float(n)) for n in range(1,11)]]*2
        interval=bootstrap(blocks,500)
        self.assertEqual(interval['lower'],2)
        self.assertEqual(interval['upper'],2)

    def test_variability_uses_sample_standard_deviation(self):
        result=stats([1,2,3])
        self.assertEqual(result['sample_stddev'],1)
        self.assertEqual(result['mean'],2)

    def test_nonfinite_observation_is_rejected(self):
        with self.assertRaises(ValueError):stats([1,float('nan')])

    def test_complete_prefill_pair_matrix_passes(self):
        result=self.analyze_fixture(*case_fixture())
        self.assertTrue(result['eligible_for_timing'],result['reasons'])
        self.assertEqual(result['verified_invocations_per_rank'],62)
        self.assertEqual(result['paired_bootstrap95']['lower'],1.25)

    def test_zero_helper_decode_is_an_explicit_unaffected_control(self):
        result=self.analyze_fixture(*case_fixture('decode'))
        self.assertTrue(result['eligible_for_timing'],result['reasons'])
        raw,context=case_fixture('decode');raw['captured_input_correctness'][0]['passed']=True
        self.assertFalse(self.analyze_fixture(raw,context)['verified'])

    def test_raw_warmup_failure_is_not_hidden_by_combined_pass_flag(self):
        raw,context=case_fixture()
        raw['blocks'][0]['attempts'][0]['warmups'][2]['observations']['fused']['ranks'][1]['correctness']['passed']=False
        self.assertFalse(self.analyze_fixture(raw,context)['verified'])

    def test_missing_capture_stop_receipt_invalidates_profile(self):
        raw,context=case_fixture(instrumented=True)
        good=self.analyze_fixture(raw,context);self.assertTrue(good['verified'],good['reasons']);self.assertFalse(good['eligible_for_timing'])
        raw['blocks'][1]['attempts'][0]['capture'].pop()
        self.assertFalse(self.analyze_fixture(raw,context)['verified'])

    def test_raw_helper_counts_and_reference_hashes_override_pass_flags(self):
        for kind in ('counts','reference'):
            raw,context=case_fixture();row=raw['blocks'][0]['attempts'][0]['measurements'][0]['observations']['fused']['ranks'][0]
            if kind=='counts':row['helper_calls']=47
            else:row['correctness'].update(sha256='e'*64,reference_sha256='e'*64)
            result=self.analyze_fixture(raw,context)
            self.assertFalse(result['verified']);self.assertIsNone(result['speedup'])

    def test_duplicate_confirmation_block_and_missing_state_check_are_rejected(self):
        for kind in ('blocks','state'):
            raw,context=case_fixture()
            if kind=='blocks':raw['blocks'][1]['block_index']=0
            else:raw['final_state'][1]['checkpoint']['verified_write_boundary_count']=61
            self.assertFalse(self.analyze_fixture(raw,context)['verified'])

    def test_native_metadata_hash_map_schema_and_drift_are_checked(self):
        for mutation in ('entry_changed','missing_key','empty_map','invalid_digest','scalar_changed','scalar_invalid',
                         'missing_region','old_scalar_schema'):
            with self.subTest(mutation=mutation):
                raw,context=case_fixture();checkpoint=raw['final_state'][0]['checkpoint']
                mapping=checkpoint['metadata_sha256'];name=next(iter(mapping))
                if mutation=='entry_changed':mapping[name]='e'*64
                elif mutation=='missing_key':mapping.pop(name)
                elif mutation=='empty_map':checkpoint['metadata_sha256']={}
                elif mutation=='invalid_digest':mapping[name]='not-a-sha256'
                elif mutation=='scalar_changed':checkpoint['scalar_metadata_sha256']='e'*64
                elif mutation=='scalar_invalid':checkpoint['scalar_metadata_sha256']={}
                elif mutation=='missing_region':checkpoint['regions'].pop()
                else:checkpoint['metadata_sha256']='a'*64
                self.assertFalse(self.analyze_fixture(raw,context)['verified'])

    def test_identical_maps_cannot_silently_omit_a_snapshot_tensor(self):
        raw,context=case_fixture()
        for row in (raw['prepared_workers'][0],raw['final_state'][0]):
            row['checkpoint']['metadata_sha256'].pop('sequence.positions')
        self.assertFalse(self.analyze_fixture(raw,context)['verified'])

    def test_negative_duration_cannot_become_an_improvement(self):
        raw,context=case_fixture();raw['blocks'][0]['attempts'][0]['measurements'][0]['observations']['fused']['ranks'][0]['wall_seconds']=-1
        self.assertFalse(self.analyze_fixture(raw,context)['verified'])

    def test_session_requires_headers_native_hashes_patch_restore_and_input_ids(self):
        for mutation in ('none','missing_manifest_header','native_hash','restore','token_ids','config'):
            with tempfile.TemporaryDirectory() as name:
                root=Path(name);verified=session_fixture(root);folder=root/'benchmark-baseline'
                if mutation=='missing_manifest_header':verified.remove(folder/'runtime.json')
                elif mutation in ('native_hash','restore','token_ids','config'):
                    filename={'native_hash':'runtime.json','restore':'summary.json','token_ids':'inputs.json','config':'config.json'}[mutation]
                    path=folder/filename;value=json.loads(path.read_text())
                    if mutation=='native_hash':value['workers'][0]['native_before_patch_evidence']['installed_sources']['model.py']['actual_sha256']='0'*64
                    elif mutation=='restore':value['patch_restoration'][0]['helper_patch']['restored']=False
                    elif mutation=='token_ids':value['token_ids'][0]+=1
                    else:value['engine_arguments']['max_model_len']=512
                    path.write_text(json.dumps(value))
                result,_=a.check_session(root,'baseline',{'verified':True},verified)
                self.assertEqual(result['verified'],mutation=='none',(mutation,result['errors']))

    def test_helper_timing_flags_and_statistics_are_recomputed(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);raw=[];checks=[];context={'folder':root,'instrumented':False,'input_captures':{}}
            verified=set()
            for rank in (0,1):
                key=str(rank)*64;context['input_captures'][rank]={key:{'capture_id':key}}
                checks.append({'rank':rank,'captures':[{'capture_id':key}]})
                observations=[{'iteration':i,'variant':v,'repeats':20,'wall_us_per_call':2. if v=='native' else 1.,
                    'cuda_event_us_per_call':1.,'compilation_events':[],'before_allocated_bytes':100,'peak_allocated_bytes':120,
                    'incremental_peak_allocated_bytes':20}
                    for i in range(10) for v in (('native','fused') if i%2==0 else ('fused','native'))]
                report={'capture_id':key,'warmup_rounds':[[{'iteration':i,'variant':v,'compilation_events':[]} for i in range(5) for v in ('native','fused')]],
                    'observations':observations,'eligible_for_helper_timing':True,
                    'summary':{v:{metric:a.stats([r[metric] for r in observations if r['variant']==v]) for metric in ('wall_us_per_call','cuda_event_us_per_call')}
                               for v in ('native','fused')}}
                row={'rank':rank,'instrumented':False,'execution_mode':'eager','applicable':True,'status':'observed','reports':[report]}
                path=root/f'helper-microbench-prefill-L256-rank-{rank}.json';path.write_text(json.dumps(row));verified.add(path);raw.append(row)
            normalized,proof=a.check_microbenchmark(raw,context,'prefill',256,verified,checks)
            self.assertTrue(proof['verified'],proof['errors']);self.assertTrue(normalized[0]['reports'][0]['eligible_for_helper_timing'])
            raw[0]['reports'][0]['observations'][0]['compilation_events']=[{'event':'compile'}]
            (root/'helper-microbench-prefill-L256-rank-0.json').write_text(json.dumps(raw[0]))
            normalized,proof=a.check_microbenchmark(raw,context,'prefill',256,verified,checks)
            self.assertFalse(proof['verified']);self.assertFalse(normalized[0]['reports'][0]['eligible_for_helper_timing'])

    def test_missing_profile_does_not_invalidate_complete_baseline_hashes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);manifested_baseline_fixture(root)
            integrity,verified=a.verify_files(root)
            self.assertFalse(integrity['verified'])
            self.assertTrue(integrity['shared']['verified'],integrity['shared']['errors'])
            self.assertTrue(integrity['sessions']['baseline']['verified'],integrity['sessions']['baseline']['errors'])
            self.assertFalse(integrity['sessions']['profile']['verified'])
            self.assertIn('Missing benchmark-profile/manifest.json',integrity['errors'])
            session,_=a.check_session(root,'baseline',integrity['sessions']['baseline'],verified)
            self.assertTrue(session['verified'],session['errors'])

    def test_missing_or_corrupt_baseline_artifacts_are_never_waived(self):
        names=('captured-inputs/rank-0/capture.pt','runtime.json','case-prefill-L256.json',
               'worker-observations/warmup.json')
        for artifact in names:
            for operation in ('missing','corrupt'):
                with self.subTest(artifact=artifact,operation=operation), tempfile.TemporaryDirectory() as name:
                    root=Path(name);manifested_baseline_fixture(root)
                    path=root/'benchmark-baseline'/artifact
                    if operation=='missing':path.unlink()
                    else:path.write_bytes(path.read_bytes()+b'corrupted')
                    integrity,verified=a.verify_files(root)
                    self.assertFalse(integrity['shared']['verified'])
                    self.assertFalse(integrity['sessions']['baseline']['verified'])
                    session,_=a.check_session(root,'baseline',integrity['sessions']['baseline'],verified)
                    self.assertFalse(session['verified'])

    def test_shared_frozen_and_root_manifest_corruption_blocks_both_sessions(self):
        for artifact in ('executed-source/experiments/relative_bias/kernel.py','cloud-worker.json','manifest.json'):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as name:
                root=Path(name);manifested_baseline_fixture(root);path=root/artifact
                if artifact=='manifest.json':
                    value=json.loads(path.read_text());value['files'][0]['sha256']='0'*64;path.write_text(json.dumps(value))
                else:path.write_bytes(path.read_bytes()+b'corrupted')
                integrity,_=a.verify_files(root)
                self.assertFalse(integrity['shared']['verified'])
                self.assertTrue(all(not v['verified'] for v in integrity['sessions'].values()))

    def test_session_manifest_cannot_omit_captured_input_or_disagree_with_root(self):
        for operation in ('omit','mismatch'):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as name:
                root=Path(name);manifested_baseline_fixture(root);path=root/'benchmark-baseline/manifest.json'
                value=json.loads(path.read_text());target=next(r for r in value['files'] if r['path'].endswith('capture.pt'))
                if operation=='omit':value['files'].remove(target)
                else:target['sha256']='0'*64
                path.write_text(json.dumps(value))
                integrity,_=a.verify_files(root)
                self.assertFalse(integrity['sessions']['baseline']['verified'])

    def test_baseline_timing_success_without_profile_cannot_admit_serving(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);manifested_baseline_fixture(root)
            def winning_case(path,*args):
                phase,length=path.stem.removeprefix('case-').split('-L')
                return {'phase':phase,'length':int(length),'eligible_for_timing':True,'speedup':1.1,
                        'blocks':[{'speedup':1.1},{'speedup':1.1}],'paired_bootstrap95':{'lower':1.05}}
            with mock.patch.object(a,'analyze_case',side_effect=winning_case):
                result=a.analyze(root)
            self.assertTrue(result['baseline_complete'])
            self.assertTrue(result['decision']['baseline_timing_gates_passed'])
            self.assertEqual(result['status'],'incomplete')
            self.assertFalse(result['profile_complete'])
            for flag in ('eager_winner','eligible_for_serving_validation','serving_validated','adopted'):
                self.assertFalse(result['decision'][flag])
            self.assertIn('Missing benchmark-profile/manifest.json',result['integrity']['errors'])


if __name__=='__main__':unittest.main()
