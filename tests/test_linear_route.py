import unittest
from a2o.scheduling.linear_route import fit_line,predict,threshold

class LinearRouteTest(unittest.TestCase):
    def test_fit_and_real_crossover(self):
        fit=fit_line([(1,12),(2,14),(8,26)])
        self.assertAlmostEqual(fit['intercept_ms'],10)
        self.assertAlmostEqual(fit['slope_ms_per_token'],2)
        def f(a,b):return dict(intercept_ms=a,slope_ms_per_token=b)
        model=dict(cpu_restore=f(2,.01),cpu_prefill=f(5,10),htp_prefill=f(30,1),htp_to_cpu=f(20,.02))
        self.assertEqual(predict(model,500,1)['route'],'cpu')
        self.assertEqual(predict(model,500,32)['route'],'htp')
        self.assertGreater(threshold(model,1000)['p_crossover'],threshold(model,100)['p_crossover'])
    def test_lower_bound_overrides_cheap_npu(self):
        def f(a,b):return dict(intercept_ms=a,slope_ms_per_token=b)
        model=dict(cpu_restore=f(0,0),cpu_prefill=f(100,1),htp_prefill=f(0,0),htp_to_cpu=f(0,0),routing_p_min=32)
        self.assertEqual(predict(model,500,31)['route'],'cpu')
        self.assertEqual(predict(model,500,32)['route'],'htp')

    def test_nonnegative_constraint(self):
        fit=fit_line([(1,10),(2,8),(3,6)])
        self.assertEqual(fit['slope_ms_per_token'],0)
        fit=fit_line([(1,1),(2,3),(3,5)])
        self.assertGreaterEqual(fit['intercept_ms'],0)
    def test_invalid_data(self):
        with self.assertRaises(ValueError):fit_line([(1,1),(1,2),(1,3)])
        with self.assertRaises(ValueError):fit_line([(1,1),(2,float('nan')),(3,3)])

if __name__=='__main__':unittest.main()
