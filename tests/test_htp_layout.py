import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from a2o.validation.htp_layout import parse_layout_line


class LayoutTelemetry(unittest.TestCase):
    def line(self, values):
        return 'profile-op MUL_MAT: ffn_up : dims : q4_0:f32:f32 : info usec 123 cycles 456 pmu ['+', '.join(str(v) for v in values)+']'

    def test_direct_hmx_layout(self):
        r=parse_layout_line(self.line([0xA2100002,65,12288,4096,2,96,512,1]))
        self.assertEqual((r['path'],r['M'],r['mc'],r['nc'],r['layout'],r['usec']),('HMX',65,96,512,1,123))

    def test_hvx_small_m(self):
        r=parse_layout_line(self.line([0xA2100801,31,12288,4096,2,0,0,0]))
        self.assertEqual((r['path'],r['reason'],r['mc']),('HVX','M_lt_32',0))

    def test_other_pmu_protocol_is_not_layout(self):
        self.assertIsNone(parse_layout_line(self.line([0xA20F0002,65,12288,4096,2,0,1,1])))
        self.assertIsNone(parse_layout_line(self.line([0]*8)))


if __name__=='__main__':unittest.main()
