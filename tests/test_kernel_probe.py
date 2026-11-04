import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.validation.kernel_probe import decode_route, parse_route_line


class KernelProbeTest(unittest.TestCase):
    def test_real_pmu_is_not_a_probe(self):
        self.assertIsNone(decode_route([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_actual_hvx_and_hmx_payloads(self):
        hvx = decode_route([0xA20F0801, 31, 2304, 128, 1, 0, 2, 16])
        hmx = decode_route([0xA20F0002, 32, 2304, 128, 1, 0, 2, 16])
        self.assertEqual((hvx['path'], hvx['reason'], hvx['M']), ('HVX', 'M_lt_32', 31))
        self.assertEqual((hmx['path'], hmx['hmx_return_code']), ('HMX', 0))

    def test_log_parser_preserves_host_and_actual_dsp_types(self):
        line = 'ggml-hex: HTP0 profile-op MUL_MAT: w x x -> y : 2048:2048 x 2048:32 -> 2048:32 : q4_0 x f32 -> f32 : 18:1152 x 4:8192 -> 4:8192 : usec 863 cycles 1377375 pmu [' + ','.join(map(str,[0xA20F0301,32,2048,2048,200,0,1,1])) + ']'
        record = parse_route_line(line)
        self.assertEqual(record['dsp_weight_type'], 200)
        self.assertEqual(record['host_types'], 'q4_0 x f32 -> f32')
        self.assertEqual(record['reason'], 'weight_type_unsupported')


if __name__ == '__main__': unittest.main()
