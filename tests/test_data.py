"""
Unit test untuk lapisan data. Tidak butuh TensorFlow, OpenCV, atau GPU.

Jalankan:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data import (
    Sample,
    collect_samples,
    deduplicate,
    discover_classes,
    stratified_split,
    summarize,
    validate_classes,
    verify_no_overlap,
)


def make_fake_dataset(root: Path, layout: dict[str, int]) -> None:
    """Bikin struktur dataset palsu: nama kelas -> jumlah gambar."""
    for label, count in layout.items():
        folder = root / label
        folder.mkdir(parents=True)
        for index in range(count):
            # isi berbeda per file supaya dedupe tidak menghapus semuanya
            (folder / f"{label}_{index}.jpg").write_bytes(
                f"{label}-{index}".encode("utf-8")
            )


class TestDiscoverClasses(unittest.TestCase):
    def test_sorted_and_dirs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_fake_dataset(root, {"Tomato_healthy": 2, "Apple_scab": 1})
            (root / "catatan.txt").write_text("bukan folder", encoding="utf-8")
            self.assertEqual(discover_classes(root), ["Apple_scab", "Tomato_healthy"])

    def test_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            discover_classes(Path("/tidak/ada/folder"))

    def test_empty_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                discover_classes(Path(tmp))


class TestCollectSamples(unittest.TestCase):
    def test_filters_extensions_and_missing_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_fake_dataset(root, {"Tomato_healthy": 2})
            (root / "Tomato_healthy" / "notes.md").write_text("x", encoding="utf-8")
            samples = collect_samples(root, ["Tomato_healthy"])
            self.assertEqual(len(samples), 2)
            self.assertTrue(all(s.label == "Tomato_healthy" for s in samples))

            with self.assertRaises(FileNotFoundError):
                collect_samples(root, ["Tidak_ada"])

    def test_empty_class_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Kelas_kosong").mkdir()
            with self.assertRaises(RuntimeError):
                collect_samples(root, ["Kelas_kosong"])


class TestDeduplicate(unittest.TestCase):
    def test_removes_identical_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            (root / "a.jpg").write_bytes(b"isi-sama")
            (root / "b.jpg").write_bytes(b"isi-sama")
            (root / "c.jpg").write_bytes(b"isi-beda")
            samples = [Sample(root / name, "X") for name in ("a.jpg", "b.jpg", "c.jpg")]
            kept, removed = deduplicate(samples)
            self.assertEqual(removed, 1)
            self.assertEqual(len(kept), 2)


class TestStratifiedSplit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        make_fake_dataset(root, {"A": 100, "B": 100})
        self.samples = collect_samples(root, ["A", "B"])
        self.labels = ["A", "B"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_overlap_and_all_used(self):
        split = stratified_split(self.samples, 0.15, 0.15, seed=42, label_order=self.labels)
        verify_no_overlap(split)
        total = len(split.train) + len(split.val) + len(split.test)
        self.assertEqual(total, len(self.samples))

    def test_deterministic_for_same_seed(self):
        first = stratified_split(self.samples, 0.15, 0.15, seed=7, label_order=self.labels)
        second = stratified_split(self.samples, 0.15, 0.15, seed=7, label_order=self.labels)
        self.assertEqual(
            [s.path for s in first.test], [s.path for s in second.test]
        )

    def test_different_seed_changes_split(self):
        first = stratified_split(self.samples, 0.15, 0.15, seed=1, label_order=self.labels)
        second = stratified_split(self.samples, 0.15, 0.15, seed=2, label_order=self.labels)
        self.assertNotEqual([s.path for s in first.test], [s.path for s in second.test])

    def test_proportions_are_reasonable(self):
        split = stratified_split(self.samples, 0.2, 0.2, seed=42, label_order=self.labels)
        self.assertAlmostEqual(len(split.val) / len(self.samples), 0.2, delta=0.03)
        self.assertAlmostEqual(len(split.test) / len(self.samples), 0.2, delta=0.03)

    def test_stratified_per_class(self):
        split = stratified_split(self.samples, 0.2, 0.2, seed=42, label_order=self.labels)
        summary = summarize(split, self.labels)
        for label in self.labels:
            self.assertAlmostEqual(summary[label]["test"] / 100, 0.2, delta=0.02)

    def test_tiny_class_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_fake_dataset(root, {"Kecil": 2, "Besar": 50})
            samples = collect_samples(root, ["Besar", "Kecil"])
            split = stratified_split(samples, 0.2, 0.2, seed=3, label_order=["Besar", "Kecil"])
            verify_no_overlap(split)
            self.assertEqual(
                len(split.train) + len(split.val) + len(split.test), len(samples)
            )

    def test_invalid_fractions_raise(self):
        with self.assertRaises(ValueError):
            stratified_split(self.samples, 0.6, 0.6, label_order=self.labels)
        with self.assertRaises(ValueError):
            stratified_split(self.samples, -0.1, 0.1, label_order=self.labels)

    def test_empty_split_detected(self):
        split = stratified_split(self.samples[:100], 0.0, 0.0, seed=1, label_order=["A"])
        with self.assertRaises(AssertionError):
            verify_no_overlap(split)


class TestClassCoverage(unittest.TestCase):
    """Ini penjaga bug yang dulu bikin 5 dari 15 kelas tidak punya penanganan."""

    def test_expected_class_count_is_fifteen(self):
        self.assertEqual(len(config.EXPECTED_CLASSES), 15)

    def test_every_expected_class_has_profile(self):
        for name in config.EXPECTED_CLASSES:
            self.assertIn(name, config.DISEASE_INFO)
            profile = config.profile_for(name)
            self.assertTrue(profile.name_id)
            self.assertTrue(profile.treatment)
            self.assertNotEqual(profile.severity, "Tidak diketahui")

    def test_validate_classes_reports_unknown(self):
        unknown, missing = validate_classes(
            ["Tomato_healthy", "Kelas_asing"], config.DISEASE_INFO
        )
        self.assertEqual(unknown, ["Kelas_asing"])
        self.assertEqual(len(missing), 14)

    def test_profile_for_unknown_returns_fallback(self):
        profile = config.profile_for("Kelas_yang_tidak_dikenal")
        self.assertEqual(profile.severity, "Tidak diketahui")
        self.assertIn("belum punya profil", profile.treatment)


if __name__ == "__main__":
    unittest.main(verbosity=2)
