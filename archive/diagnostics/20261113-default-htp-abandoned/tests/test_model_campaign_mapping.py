"""映射实验必须传递配置，并拒绝健康端点遗漏或错报双会话。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from a2o.profiling.model_cost_campaign import Campaign,campaign_jobs

class MappingConfig(unittest.TestCase):
    def test_mapping_conditions_preserve_schedule_and_reach_launcher(self):
        base=json.loads((ROOT/'configs/qwen3-8b-key-costs.json').read_text())
        for label,sessions,mbuf in [('dual-htp',2,1024),('htp-mbuf128',1,128)]:
            cfg=json.loads((ROOT/f'configs/qwen3-8b-kv-{label}.json').read_text())
            self.assertEqual(campaign_jobs(base),campaign_jobs(cfg))
            with tempfile.TemporaryDirectory() as tmp:
                c=Campaign(cfg,Path(tmp),Path('/private/agent'))
                with patch('a2o.profiling.model_cost_campaign.subprocess.run') as run:
                    c.control('start')
                args=run.call_args.args[0]
                for key,value in [('--htp-sessions',str(sessions)),('--htp-max-buffer-mib',str(mbuf)),('--kv-offload','1'),('--flash-attention','0')]:
                    self.assertEqual(args[args.index(key)+1],value)
                health=dict(condition='route',model=cfg['model_id'],cpu_threads=6,context_size=4608,htp_kv_offload=True,flash_attention=False,single_resident=True)
                with patch.object(c,'health',return_value=health):
                    with self.assertRaises(RuntimeError):c.check_identity()
                health.update(htp_sessions=sessions,htp_max_buffer_mib=mbuf)
                with patch.object(c,'health',return_value=health):c.check_identity()

if __name__=='__main__':unittest.main()
