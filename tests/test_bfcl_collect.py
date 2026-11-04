import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from a2o.workloads.bfcl import common_prefix, parse_calls, run_case, load_entries, _tool_ok


class ContractTests(unittest.TestCase):
    def test_literal_calls_and_rejection(self):
        self.assertEqual(parse_calls("[cat(file_name='a.txt')]",{'cat'})[0]['kwargs'],{'file_name':'a.txt'})
        for bad in ["[cat(file_name=__import__('os').getcwd())]", "[os.system('id')]",
                    "[cat(**{'file_name':'a'})]", "[unknown()]", "[x for x in []]"]:
            with self.subTest(bad=bad), self.assertRaises((ValueError,TypeError)):
                parse_calls(bad,{'cat'})
        self.assertEqual(parse_calls('Please provide a filename.',{'cat'}),[])

    def test_prefix_is_measured(self):
        self.assertEqual(common_prefix([1,2,3],[1,2,4,5]),2)
        self.assertEqual(common_prefix([],[1]),0)

    def test_tool_error_is_not_success(self):
        self.assertFalse(_tool_ok({'error':None,'result':'{"error":"not found"}'}))
        self.assertFalse(_tool_ok({'error':'timeout','result':''}))
        self.assertTrue(_tool_ok({'error':None,'result':'{"file_content":"ok"}'}))

    def test_multiturn_sandbox_and_time_links(self):
        root=Path(os.environ.get('BFCL_REFERENCE',str(ROOT.parent/'shared/references/gorilla-bfcl/berkeley-function-call-leaderboard')))
        if not root.exists(): self.skipTest('shared BFCL reference not installed')
        sys.path.insert(0,str(root))
        cfg={'categories':['multi_turn_base'],'port':18765,'tool_timeout_s':10,
             'request_timeout_s':5,'max_steps':20,'max_tokens':512,'seed':1,
             'context_per_slot':1000000,'bfcl_root':str(root)}
        entry=load_entries(root,cfg)[0]
        truth=json.loads((root/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines()[0])['ground_truth']
        replies=[]
        for calls in truth:
            replies.extend(['['+', '.join(calls)+']','[]'])
        iterator=iter(replies)
        def fake_post(base,route,data,timeout=300):
            if route=='/tokenize': return {'tokens':list(data['content'].encode())}
            text=next(iterator)
            return {'content':text,'tokens':list(text.encode()),'stop_type':'eos','timings':{}}
        with tempfile.TemporaryDirectory() as temp, patch('a2o.workloads.bfcl.post',side_effect=fake_post):
            result=run_case(entry,cfg,'test',0,Path(temp))
            self.assertTrue(result['trace_complete'],result)
            self.assertTrue(result['quality']['valid'],result)
            events=[json.loads(s) for s in (Path(temp)/'test/multi_turn_base_0/events.jsonl').read_text().splitlines()]
            tools=[e for e in events if e['kind']=='tool']
            self.assertEqual(len(tools),sum(map(len,truth)))
            for e in tools:
                self.assertLessEqual(e['issued_ns'],e['execution_start_ns'])
                self.assertLessEqual(e['execution_complete_ns'],e['result_available_ns'])
            self.assertEqual(sum(e['kind']=='tool_next_decode' for e in events),len(tools))


if __name__=='__main__': unittest.main()
