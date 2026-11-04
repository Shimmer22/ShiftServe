"""防止把缺少机制/状态证据或不完整的对照写成成功结果。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kv_prepack_analysis',ROOT/'scripts/analyze/kv_prepack.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class PrepackAnalysisTest(unittest.TestCase):
    def fixture(self,out):
        (out/'device').mkdir();(out/'device/agent').write_bytes(b'binary');(out/'P32.txt').write_bytes(b'fixture')
        cfg={'new_tokens':[32],'history_tokens':4096,'gap_ms':[0],'repetitions':3}
        (out/'manifest.json').write_text(json.dumps({'config':cfg,'binary_sha256':hashlib.sha256(b'binary').hexdigest()}))
        (out/'completion.json').write_text('{"passed":true}');(out/'candidate-qualification.json').write_text('{"32":true}')
        rows=[]
        for rep in range(3):
            for route in ('CC','HC'): rows.append(self.row(len(rows),'calibration',rep,route,'after_return',route=='HC' and rep>0))
        for rep in range(3):
            for strategy in ('after_return','always'):
                for pack in (False,True): rows.append(self.row(len(rows),'measurement',rep,'HC',strategy,pack))
        self.save(out,rows);return rows

    def row(self,i,phase,rep,route,strategy,pack):
        size=4096*256*2*72 if pack else 0
        return dict(id=i,phase=phase,repetition=rep,P=32,D=6,route=route,strategy=strategy,kv_prepack_enabled=pack,
            gap_requested_ms=0,passed=True,response_check={'passed':True},eog=True,handoff_bytes=1,
            output='{"action":"wait"}',generated_ids=[1,2],numerical_passed=True,
            htp_baseline_compared=pack,htp_baseline_state_compared=pack,htp_baseline_nrmse=0,
            kv_prepack_entries=72 if pack else 0,kv_prepack_cache_bytes=size,kv_prepack_k_bytes=size//2,
            kv_prepack_v_bytes=size//2,kv_prepack_k_hits=72 if pack else 0,kv_prepack_v_hits=72 if pack else 0,
            response_compute_completed_ms=300,response_completed_ms=302,http_issued_ms=0,http_returned_ms=100,
            prepared_early=strategy=='always',ready_at_return=strategy=='always',prepare_started_ms=1,prepare_completed_ms=80,
            response_after_return_ms=200-10*pack,tool_to_response_ms=300,actual_gap_ms=100,prepare_ms=79,
            kv_prepack_ms=10*pack,prefill_ms=100,handoff_checked_ms=10,decode_ms=90,wait_or_prepare_after_return_ms=0)

    def save(self,out,rows):
        write=lambda name,data:(out/name).write_text(''.join(json.dumps(r)+'\n' for r in data))
        write('checked-rows.jsonl',rows);write('device/trials.jsonl',rows)
        write('http-server.jsonl',[{'id':r['id'],'P':32,'gap_requested_ms':0,'body_sha256':hashlib.sha256(b'fixture').hexdigest()} for r in rows])
        write('native-events.jsonl',[{'event':'response_published','turn':r['id'],'monotonic_ms':301} for r in rows])

    def test_complete_mechanism_and_counterfactual(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);self.fixture(out);result=module.analyze(out)
            self.assertTrue(result['passed']);self.assertEqual(result['comparisons'][0]['incremental_prepack_gain_ms'],10)

    def test_reject_missing_final_state_comparison(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);rows=self.fixture(out);rows[3]['htp_baseline_state_compared']=False;self.save(out,rows)
            with self.assertRaises(AssertionError):module.analyze(out)

    def test_reject_partial_kv_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);rows=self.fixture(out);rows[-1]['kv_prepack_v_bytes']=0;self.save(out,rows)
            with self.assertRaises(AssertionError):module.analyze(out)

    def test_reject_missing_kv_hits(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);rows=self.fixture(out);rows[-1]['kv_prepack_v_hits']=0;self.save(out,rows)
            with self.assertRaises(AssertionError):module.analyze(out)

    def test_reject_missing_early_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);rows=self.fixture(out)
            for row in rows:row['prepared_early']=False
            self.save(out,rows)
            with self.assertRaises(AssertionError):module.analyze(out)

    def test_reject_missing_requested_release(self):
        for key in ('htp_kv_release_before_decode','release_calibration_state'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                out=Path(d);self.fixture(out)
                manifest=json.loads((out/'manifest.json').read_text());manifest['config'][key]=True
                (out/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(AssertionError):module.analyze(out)

    def test_reject_missing_replay(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);rows=self.fixture(out);self.save(out,rows[:-1])
            with self.assertRaises(AssertionError):module.analyze(out)

if __name__=='__main__':unittest.main()
