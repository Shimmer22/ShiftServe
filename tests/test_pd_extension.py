import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.profiling.pd_extension import merge_runs


class ExtensionMerge(unittest.TestCase):
    def fixtures(self, root, wrong_threads=False):
        folders = [root / 'base', root / 'extra', root / 'out']
        for folder in folders:
            folder.mkdir()
        for folder, points in zip(folders, [list(range(24, 65)) + [128], [66, 68, 72, 80, 96]]):
            config = dict(H=4096, rounds=10, cpu_threads=6, context_size=8192, D=[], P=points)
            (folder / 'plan.json').write_text(json.dumps(dict(config=config, binary_sha256='same', source_sha256='same')))
            (folder / 'status.json').write_text(json.dumps(dict(phase='complete')))
            with (folder / 'stage-times.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=['round', 'stage', 'backend', 'H', 'P', 'n_threads', 'n_threads_batch'])
                writer.writeheader()
                for p in points:
                    for rep in range(10):
                        for backend in ['cpu', 'htp']:
                            threads = 6 if backend == 'cpu' else 4
                            if wrong_threads and folder == folders[1]:threads = 4
                            writer.writerow(dict(round=rep, stage='prefill', backend=backend, H=4096, P=p,
                                                 n_threads=threads, n_threads_batch=threads))
        return folders

    def test_merge_47_points_ten_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            base, extra, out = self.fixtures(Path(directory))
            merge_runs(base, extra, out)
            self.assertEqual(json.loads((out / 'summary.json').read_text())['stage_rows'], 940)

    def test_thread_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base, extra, out = self.fixtures(Path(directory), wrong_threads=True)
            with self.assertRaises(AssertionError):
                merge_runs(base, extra, out)


if __name__ == '__main__':
    unittest.main()
