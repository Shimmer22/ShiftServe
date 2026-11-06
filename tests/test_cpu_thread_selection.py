"""线程选择使用独立、完整的重复组和等权形状，不能逐点挑最快。"""
import unittest
from a2o.profiling.cpu_threads import select_threads


class CpuThreadSelectionTest(unittest.TestCase):
    def rows(self):
        result=[]
        for p,base,ratio in [(16,100,.8),(64,1000,1.1)]:
            for t in [4,6]:
                for rep in range(3):result.append(dict(phase='measurement',H=2048,P=p,cpu_prefill_threads=t,prefill_ms=base*(ratio if t==6 else 1),repetition=rep))
        return result

    def test_shapes_are_equal_weighted_not_dominated_by_largest_input(self):
        result=select_threads(self.rows(),[4,6])
        self.assertEqual(result['best_threads'],6)
        self.assertAlmostEqual(result['scores'][6],(.8*1.1)**.5)
        self.assertEqual(len(result['cells']),4)

    def test_incomplete_repeats_cannot_select_a_success_subset(self):
        with self.assertRaises(ValueError):select_threads(self.rows()[:-1],[4,6])

    def test_numerical_calibration_is_not_performance_selection_data(self):
        rows=self.rows()+[dict(phase='calibration',H=2048,P=16,cpu_prefill_threads=4,prefill_ms=.01)]
        self.assertEqual(select_threads(rows,[4,6]),select_threads(self.rows(),[4,6]))
