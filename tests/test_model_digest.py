import tempfile
import unittest
from pathlib import Path

from a2o.runtime.model_digest import verify_model_digest


class ModelDigestTest(unittest.TestCase):
    def test_reuse_only_for_same_file_on_same_boot(self):
        identity = ["boot-a\n1:2:3:mtime:ctime\n"]
        hashes = []

        def shell(command):
            if command.startswith("sha256sum"):
                hashes.append(command)
                return "expected  model\n"
            return identity[0]

        with tempfile.TemporaryDirectory() as tmp:
            def verify(force=False):
                return verify_model_digest(shell, "phone", "/models/test gguf", "expected",
                                           force=force, cache_dir=Path(tmp))["source"]

            self.assertEqual(verify(), "full_sha256")
            self.assertEqual(verify(), "cached_stat")
            identity[0] = "boot-a\n1:2:3:mtime:new-ctime\n"
            self.assertEqual(verify(), "full_sha256")
            identity[0] = "boot-b\n1:2:3:mtime:new-ctime\n"
            self.assertEqual(verify(), "full_sha256")
            self.assertEqual(verify(force=True), "full_sha256")
            self.assertEqual(len(hashes), 4)

    def test_changed_model_during_hash_does_not_publish_receipt(self):
        outputs = iter(("boot\noriginal\n", "expected  model\n", "boot\nchanged\n"))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "校验期间变更"):
                verify_model_digest(lambda _: next(outputs), "phone", "/model", "expected",
                                    cache_dir=Path(tmp))
            self.assertFalse(list(Path(tmp).iterdir()))
