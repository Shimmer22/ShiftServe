"""主图Oracle区分计时起点，并保留固定提前HC与状态匹配约束。"""
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('observation_plot',ROOT/'scripts/plot/observations.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class ObservationOracleTest(unittest.TestCase):
    def test_non_atomic_clock_read_skew_keeps_original_data(self):
        row=dict(actual_gap_ms=100,active_actual_gap_ms=100.026,tool_to_response_ms=1000,
                 active_tool_to_response_ms=1000.025,suspend_during_tool_to_response_ms=-.025)
        before=dict(row);m.check_dual_clock(row)
        self.assertEqual(row,before)

    def test_material_clock_discrepancy_still_fails(self):
        row=dict(actual_gap_ms=100,active_actual_gap_ms=100.5,tool_to_response_ms=1000,
                 active_tool_to_response_ms=1000.5,suspend_during_tool_to_response_ms=-.5)
        with self.assertRaises(ValueError):m.check_dual_clock(row)

    def test_clock_accounting_identity_retains_strict_tolerance(self):
        row=dict(actual_gap_ms=100,active_actual_gap_ms=100,tool_to_response_ms=1000,
                 active_tool_to_response_ms=1000,suspend_during_tool_to_response_ms=.05)
        with self.assertRaises(ValueError):m.check_dual_clock(row)

    def rows(self):
        rows=[]
        for strategy in ['after_return','always','cost_aware']:
            for requested in ['CC','HC','router']:
                route='CC' if requested=='router' else requested
                response=300;gap=10
                if (strategy,requested)==('always','HC'):response=40
                if (strategy,requested)==('always','router'):response=250
                if (strategy,requested)==('cost_aware','router'):response=200
                if (strategy,requested)==('cost_aware','CC'):response=210
                if (strategy,requested)==('cost_aware','HC'):response=100;gap=1000
                for rep in range(3):
                    rows.append(dict(id=len(rows),phase='measurement',H=2048,P=32,gap_requested_ms=0,
                        strategy=strategy,requested_route=requested,route=route,repetition=rep,
                        ready_at_return=strategy!='after_return',prepared_early=strategy!='after_return',
                        response_after_return_ms=response,actual_gap_ms=gap,tool_to_response_ms=response+gap))
        return rows

    def test_routing_uses_return_time_and_joint_includes_fixed_early_hc(self):
        result=m.oracle_regret(self.rows(),2048,32,0,True)
        self.assertEqual(result['routing_regret_ms'],100)
        self.assertEqual(result['joint_empirical_gap_ms'],160)
        self.assertEqual(result['joint_oracle']['strategy'],'always')
        self.assertEqual(result['joint_oracle']['requested_route'],'HC')
        self.assertEqual(len(result['joint_oracle']['trial_ids']),3)
        self.assertEqual(len(result['joint_candidates']),9)

    def test_unmatched_state_is_null_not_false_zero(self):
        rows=self.rows()
        next(r for r in rows if (r['strategy'],r['requested_route'])==('cost_aware','HC'))['prepared_early']=False
        result=m.oracle_regret(rows,2048,32,0,True)
        self.assertIsNone(result['routing_regret_ms'])
        self.assertFalse(result['matched'])

    def test_unqualified_hc_does_not_imply_optimality(self):
        result=m.oracle_regret(self.rows(),2048,32,0,False)
        self.assertIsNone(result['routing_regret_ms'])
        self.assertIsNone(result['joint_empirical_gap_ms'])
        self.assertTrue(all(c['requested_route']!='HC' for c in result['joint_candidates']))
