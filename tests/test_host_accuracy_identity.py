import unittest

from a2o.profiling.host_accuracy import run_metadata, same_run, select_tasks


class HostAccuracyIdentityTest(unittest.TestCase):
    def test_smoke_result_is_not_a_full_campaign_cache_hit(self):
        config = {"context_size": 40960}
        runtime = {"binary_sha256": "one", "kv_layer_stride": 4}
        smoke = run_metadata("v2", config, "full_kv", 3, [], runtime)
        full = run_metadata("v2", config, "full_kv", 0, [], runtime)
        self.assertFalse(same_run(smoke, full))
        self.assertFalse(same_run(full, run_metadata("v2", config, "full_kv", 0, [],
                                                    dict(runtime, binary_sha256="two"))))
        self.assertFalse(same_run(full, run_metadata("v2", config, "full_kv", 0, ["long"], runtime)))

    def test_task_selection_preserves_benchmark_order_and_requires_known_ids(self):
        tasks = [{"id": "one"}, {"id": "long"}, {"id": "three"}]
        self.assertEqual(select_tasks(tasks, lambda task: task["id"], ["three", "one"], 0),
                         [tasks[0], tasks[2]])
        with self.assertRaisesRegex(ValueError, "不在当前评测题集"):
            select_tasks(tasks, lambda task: task["id"], ["wrong"], 0)
