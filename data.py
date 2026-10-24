"""
Pembentukan dataset: penemuan kelas, deduplikasi, split tiga arah, dan
pembuatan tf.data.Dataset.

Kenapa modul terpisah:

1. Bagian split dan dedupe ditulis murni Python dan tidak meng-import
   TensorFlow, sehingga bisa dites tanpa memasang TensorFlow
   (lihat tests/test_data.py).
2. Ketiga hal yang dulu salah sekarang ditangani di satu tempat:
   - jumlah kelas dibaca dari isi folder, bukan konstanta hardcoded;
   - gambar duplikat dibuang sebelum split, supaya tidak bocor ke train dan test;
   - split tiga arah dengan seed tetap, deterministik, dan tidak saling tumpang tindih.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str


@dataclass(frozen=True)
class Split:
    train: list[Sample]
    val: list[Sample]
    test: list[Sample]

    def as_dict(self) -> dict[str, list[Sample]]:
        return {"train": self.train, "val": self.val, "test": self.test}


# ----------------------------------------------------------------- penemuan


def discover_classes(dataset_dir: Path) -> list[str]:
    """Ambil nama kelas dari subfolder dataset, terurut dan deterministik."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan: {dataset_dir}\n"
            "Unduh PlantVillage dari Kaggle lalu ekstrak sehingga strukturnya jadi:\n"
            "  dataset/PlantVillage/Tomato_healthy/\n"
            "  dataset/PlantVillage/Tomato_Late_blight/\n"
            "  ..."
        )
    classes = sorted(
        p.name
        for p in dataset_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not classes:
        raise RuntimeError(f"Tidak ada subfolder kelas di {dataset_dir}")
    return classes


def collect_samples(dataset_dir: Path, class_names: list[str]) -> list[Sample]:
    """Kumpulkan semua file gambar untuk kelas yang diminta."""
    samples: list[Sample] = []
    for label in class_names:
        folder = dataset_dir / label
        if not folder.is_dir():
            raise FileNotFoundError(f"Kelas '{label}' tidak ada di {dataset_dir}")
        files = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise RuntimeError(f"Folder kelas '{label}' tidak berisi gambar")
        samples.extend(Sample(path=p, label=label) for p in files)
    return samples


def validate_classes(
    discovered: list[str], profiles: dict[str, object]
) -> tuple[list[str], list[str]]:
    """Pastikan tidak ada kelas yang tidak punya profil penanganan.

    Mengembalikan (unknown, missing):
      unknown -> ada di dataset tapi belum ada di DISEASE_INFO  (harus gagal)
      missing -> ada di DISEASE_INFO tapi tidak ada di dataset  (cuma peringatan)
    """
    unknown = [c for c in discovered if c not in profiles]
    missing = [c for c in profiles if c not in discovered]
    return unknown, missing


# ----------------------------------------------------------------- deduplikasi


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate(samples: list[Sample]) -> tuple[list[Sample], int]:
    """Buang gambar dengan isi byte identik.

    PlantVillage berisi banyak gambar yang benar-benar sama persis. Tanpa
    deduplikasi, salinan yang sama bisa jatuh ke train dan test sekaligus, dan
    akurasi test jadi tidak bermakna.
    """
    seen: set[str] = set()
    kept: list[Sample] = []
    for sample in samples:
        digest = file_digest(sample.path)
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(sample)
    return kept, len(samples) - len(kept)


# ----------------------------------------------------------------- split


def stratified_split(
    samples: list[Sample],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    label_order: list[str] | None = None,
) -> Split:
    """Split per kelas (stratified), deterministik untuk seed yang sama.

    Per kelas, sampel diacak dengan seed tetap lalu dibagi ke test, val, train.
    Urutan iterasi kelas memakai `label_order` bila diberikan, supaya hasilnya
    tidak bergantung pada urutan dict.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction harus di [0, 1), dapat {val_fraction}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction harus di [0, 1), dapat {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction harus kurang dari 1")

    buckets: dict[str, list[Sample]] = {}
    for sample in samples:
        buckets.setdefault(sample.label, []).append(sample)

    order = label_order if label_order is not None else sorted(buckets)
    missing = [label for label in order if label not in buckets]
    if missing:
        raise ValueError(f"label_order memuat kelas tanpa sampel: {missing}")

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []

    for label in order:
        items = list(buckets[label])
        rng.shuffle(items)
        total = len(items)

        n_test = int(round(total * test_fraction))
        n_val = int(round(total * val_fraction))

        # Tiap split minimal dapat satu sampel kalau kelasnya cukup besar.
        # Aturan ini hanya berlaku kalau fraksinya memang diminta > 0, supaya
        # --val-fraction 0 benar-benar menghasilkan val kosong.
        if total >= 3:
            if test_fraction > 0:
                n_test = max(1, n_test)
            if val_fraction > 0:
                n_val = max(1, n_val)

        # Jangan sampai seluruh kelas masuk ke val+test.
        while n_test + n_val >= total and (n_test > 0 or n_val > 0):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            else:
                n_val -= 1

        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    return Split(train=train, val=val, test=test)


def verify_no_overlap(split: Split) -> None:
    """Pastikan tidak ada path yang muncul di lebih dari satu split."""
    seen: dict[Path, str] = {}
    for part_name, part in split.as_dict().items():
        for sample in part:
            previous = seen.get(sample.path)
            if previous is not None:
                raise AssertionError(
                    f"Kebocoran data: {sample.path} ada di '{previous}' dan '{part_name}'"
                )
            seen[sample.path] = part_name
    if not split.train or not split.val or not split.test:
        raise AssertionError("Setiap split harus berisi minimal satu sampel")


def summarize(split: Split, label_order: list[str]) -> dict[str, dict[str, int]]:
    """Hitung jumlah sampel per kelas per split, untuk log dan metadata."""
    summary: dict[str, dict[str, int]] = {
        label: {"train": 0, "val": 0, "test": 0} for label in label_order
    }
    for part_name, part in split.as_dict().items():
        for sample in part:
            summary[sample.label][part_name] += 1
    return summary


# ----------------------------------------------------------------- tf.data


def make_dataset(
    samples: list[Sample],
    class_names: list[str],
    *,
    img_size: tuple[int, int],
    batch_size: int,
    preprocess: str,
    training: bool = False,
    augment=None,
    shuffle: bool = False,
    seed: int = 42,
):
    """Bangun tf.data.Dataset dari daftar Sample.

    Import TensorFlow dilakukan di dalam fungsi supaya modul ini tetap bisa
    diimpor di lingkungan tanpa TensorFlow (dipakai oleh unit test).
    """
    import tensorflow as tf

    if not samples:
        raise ValueError("Tidak ada sampel untuk dibuat dataset")

    class_index = {name: i for i, name in enumerate(class_names)}
    unknown = sorted({s.label for s in samples if s.label not in class_index})
    if unknown:
        raise ValueError(f"Sampel dengan label tidak dikenal: {unknown}")

    paths = [str(s.path) for s in samples]
    labels = [class_index[s.label] for s in samples]

    def preprocess_image(image):
        if preprocess == "mobilenet_v2":
            return tf.keras.applications.mobilenet_v2.preprocess_input(image)
        if preprocess == "rescale":
            return image / 255.0
        raise ValueError(f"Mode preprocess tidak dikenal: {preprocess}")

    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, img_size, method="bilinear", antialias=True)
        image = tf.cast(image, tf.float32)
        if training and augment is not None:
            # Layer augmentasi Keras paling aman dipanggil dengan dimensi batch.
            image = augment(tf.expand_dims(image, axis=0), training=True)[0]
        return preprocess_image(image), label

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        buffer = min(len(paths), 2048)
        dataset = dataset.shuffle(buffer, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset
