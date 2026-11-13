"""实际采样配置必须保持逐点十次、5/5先后及连续decode三轮。"""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from a2o.profiling.model_cost_campaign import campaign_jobs,paired_order


class SamplingSchedule(unittest.TestCase):
    def test_real_config_repetition_pairing_and_seed(self):
        cfg=json.loads((ROOT/'configs/qwen3-8b-key-costs.json').read_text())
        jobs=campaign_jobs(cfg)
        self.assertEqual(jobs,campaign_jobs(cfg))
        self.assertEqual(len(cfg['P']),47)
        self.assertEqual(len(jobs),473)
        for p in cfg['P']:
            group=[j for j in jobs if j[1]=='prefill' and j[2]==p]
            self.assertEqual(len(group),10)
            self.assertEqual(sum(paired_order(cfg,*j[:3])[0]=='cpu' for j in group),5)
        for rep in range(10):
            actual=[j[2] for j in jobs if j[0]==rep and j[1]=='prefill']
            self.assertEqual(sorted(actual),sorted(cfg['P']))
            self.assertNotEqual(actual,cfg['P'])
        decode=[j for j in jobs if j[1]=='decode']
        self.assertEqual(len(decode),3)
        self.assertTrue(all(j[3]==cfg['D'] and j[3][-1]==320 for j in decode))


if __name__=='__main__':
    unittest.main()
