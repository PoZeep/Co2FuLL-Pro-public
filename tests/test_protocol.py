"""Unit tests for Co²FuLL-Pro."""

import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

from co2full.assets import unpack
from co2full.io import write_json
from co2full.pipeline import check_seal, seal
from co2full.reporting import metrics
from co2full.runtime import config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_cases import score


class ProtocolTests(unittest.TestCase):
    def test_missing_candidate_prediction(self):
        with self.assertRaises(ValueError):
            metrics({"a": 1, "b": 0}, {"a": 1})

    def test_full_pool_recall(self):
        result = metrics({"a": 1, "b": 1, "c": 0}, {"a": 1, "b": 0, "c": 1}, retrieval_misses=1)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 2))
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertAlmostEqual(result["f2"], 5 / 14)
        with self.assertRaises(ValueError):
            metrics({"a": 1}, {"a": 1}, retrieval_misses=-1)

    def test_case_scoring_keeps_search_pools_separate(self):
        cases = [
            dict(
                case_id="first",
                positives=["a", "b", "c"],
                retrieved=["a", "c", "d", "e"],
                accepted=["a", "d", "e"],
                excluded=["e"],
            ),
            dict(case_id="second", positives=["d"], retrieved=["d"], accepted=["d"]),
        ]
        result = score(cases)
        self.assertEqual(
            (result["tp"], result["fp"], result["retrieval_fn"], result["verification_fn"]), (2, 1, 1, 1)
        )

    def test_config_preserves_environment_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            exe = root / "python"
            exe.symlink_to(sys.executable)
            path = root / "config.json"
            write_json(path, dict(schema="co2full-machine-v1", python="python"))
            self.assertEqual(config(path)["python"], str(exe))

    def test_unsafe_asset_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name in enumerate(("../outside", "link")):
                path = root / f"input-{i}.tar"
                with tarfile.open(path, "w") as archive:
                    member = tarfile.TarInfo(name)
                    if i:
                        member.type = tarfile.SYMTYPE
                        member.linkname = "../outside"
                        archive.addfile(member)
                    else:
                        member.size = 1
                        archive.addfile(member, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    unpack(path, root / f"assets-{i}")
                self.assertFalse((root / "outside").exists())

    def test_seal_detects_same_size_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "value"
            output.write_text("one")
            seal(root, 1, [output])
            output.write_text("two")
            with self.assertRaises(RuntimeError):
                check_seal(root, 1)


if __name__ == "__main__":
    unittest.main()
